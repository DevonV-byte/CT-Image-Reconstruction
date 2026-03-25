# Entry point for the CT image reconstruction pipeline.
# Loads sample images and runs the selected reconstruction algorithm via CLI argument.
# Supports Direct Fourier Reconstruction, Filtered Backprojection, fan-beam rebinning,
# sinogram generation, and sinogram coverage detection.
#
# Created: 2024-10-14
# Updated: 2026-03-25
# Author: Devon Vanaenrode

# --- Imports ---
import argparse
import cv2
import numpy as np
import math

import scipy
from scipy.fft import fft, ifft, fftshift, ifftshift, ifft2
import matplotlib.pyplot as plt
from skimage.transform import rotate
from skimage.transform import resize

# --- Main loop ---

def main():
    parser = argparse.ArgumentParser(description="CT Image Reconstruction Pipeline")
    parser.add_argument(
        "--mode",
        choices=["fourier", "fbp", "rebin", "sinogram", "detect", "ramlak"],
        required=True,
        help=(
            "fourier  – Direct Fourier Reconstruction (Fourier Slice Theorem)\n"
            "fbp      – Filtered Backprojection\n"
            "rebin    – Rebin divergent (fan-beam) sinogram to parallel-beam\n"
            "sinogram – Generate a sinogram from the Shepp-Logan phantom\n"
            "detect   – Detect 180° vs 360° sinogram coverage\n"
            "ramlak   – Filtered Backprojection with RAM-LAK filter"
        ),
    )
    args = parser.parse_args()

    target  = cv2.imread("Samples/lotus.png",          cv2.IMREAD_GRAYSCALE)
    sinogram = cv2.imread("Samples/lotus_parallel.png", cv2.IMREAD_GRAYSCALE)
    div_sino = cv2.imread("Samples/lotus_divergent.png", cv2.IMREAD_GRAYSCALE)

    if args.mode == "fourier":
        func3_1(target, sinogram, 3)

    elif args.mode == "fbp":
        func3_2(target, sinogram, 1.1)

    elif args.mode == "rebin":
        func3_3(div_sino, sinogram)

    elif args.mode == "sinogram":
        target_sl = cv2.imread("Samples/Shepp_Logan.png",     cv2.IMREAD_GRAYSCALE)
        sino_sl   = cv2.imread("Samples/Shepp_Logan_Sino.png", cv2.IMREAD_GRAYSCALE)
        func3_4(target_sl, sino_sl)

    elif args.mode == "detect":
        func3_5_1(sinogram)

    elif args.mode == "ramlak":
        func3_2(target, sinogram, 1.1, ram_lak_filter=True)


# --- Helpers ---

def CreateSinogram(image, max_angle, num_angles, rotate=False):
    angles = np.linspace(0, max_angle, num_angles)
    height, width = image.shape
    center = (width - 1) / 2.0, (height - 1) / 2.0

    sinogram = np.zeros((len(angles), max(height, width)))

    for i, angle in enumerate(angles):
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
        ax = 1 if rotate else 0
        sinogram[i, :] = np.sum(rotated_image, axis=ax)
    
    # normalize
    sinogram = (sinogram - sinogram.min()) / (sinogram.max() - sinogram.min())
    return sinogram

def Rebinning(div_sinogram, max_angle=360, FOD=540, FDD=630, sensor_width=120, rotate=False):
    print("Rebinning...")
    if rotate:
        div_sinogram = cv2.rotate(div_sinogram, cv2.ROTATE_90_CLOCKWISE)
    
    num_angles, num_detectors = div_sinogram.shape
    parallel_sinogram = np.zeros((num_angles, num_detectors))
    all_beta = np.linspace(0 / 2, max_angle * (math.pi / 180), num_angles)
    detector_distance = sensor_width / num_detectors
    half_fan_angles = []
    for d in range(1, int(num_detectors / 2) + 1):
        half_fan_angles.append(math.atan((d * detector_distance) / FDD))

    first_half_angles = [ -x for x in half_fan_angles]
    first_half_angles.reverse()
    first_half_angles.extend(half_fan_angles)
    fan_angles = first_half_angles

    for i, theta in enumerate(all_beta):
        if (i % 90 == 0):
            print("{} %".format(i/len(all_beta) * 100))
        row = [None] * num_detectors
        for ii, detector in enumerate(range(num_detectors)):
            s = (detector * detector_distance) - sensor_width / 2
            j = - math.asin(s / math.sqrt((FOD * FOD) - (s * s)))
            beta = theta - j

            # interbeta gives div_sinogram y value
            interbeta = np.interp(beta, all_beta, range(0, num_angles))
            interj = np.interp(j, fan_angles, range(0, num_detectors))
            b1 = math.floor(interbeta)
            b2 = math.ceil(interbeta)
            j1 = math.floor(interj)
            j2 = math.ceil(interj)
            interp = interpolate2d(interbeta, interj, b1, b2, j1, j2, div_sinogram)
            row[num_detectors - ii - 1] = interp
        parallel_sinogram[i][:] = row        

    if (rotate):
        parallel_sinogram = cv2.rotate(parallel_sinogram, cv2.ROTATE_90_COUNTERCLOCKWISE)

    cv2.imwrite("rebinned_image.jpg", parallel_sinogram)
    # normalize
    parallel_sinogram = (parallel_sinogram - parallel_sinogram.min()) / (parallel_sinogram.max() - parallel_sinogram.min())
    
    print("Done with rebinning.")

    return parallel_sinogram


def backProject(sinogram, angular_range=180.0):
    laminogram = np.zeros((sinogram.shape[1], sinogram.shape[1]))
    dTheta = angular_range / sinogram.shape[0]

    for i in range(sinogram.shape[0]):
        print(f"backprojecting: {(i + 1) * dTheta}/{angular_range}", end="\r")
        temp = np.tile(sinogram[i],(sinogram.shape[1],1))
        temp = rotate(np.absolute(temp), dTheta*i)
        laminogram += temp
    print()
    # normalize the image
    laminogram = (laminogram - laminogram.min()) / (laminogram.max() - laminogram.min())
    return laminogram

def Filter(ffts, ramp):
    plt.subplot(3, 2, 3)
    plt.title("Ramp filter")
    plt.plot(ramp)
    plt.gray()
    return ffts * ramp

def FB(sinogram, N, S, ramp):
    frequency_domain_sinogram = fft(sinogram, axis=1)

    filtered_frequency_domain_sinogram = Filter(frequency_domain_sinogram, ramp)
    filtered_spatial_domain_sinogram = ifft(filtered_frequency_domain_sinogram, axis=1)
    reconstructed_image = backProject(filtered_spatial_domain_sinogram)

    window1d = np.abs(np.hamming(reconstructed_image.shape[0]))
    window2d = np.sqrt(np.outer(window1d,window1d))
    hamming = reconstructed_image * window2d

    return hamming

def interpolate2d(beta, j, b1, b2, j1, j2, div_image):
    # surrounding points
    f11 = div_image[b1][j1]
    f21 = div_image[b2][j1]
    f12 = div_image[b1][j2]
    f22 = div_image[b2][j2]

    if ((b2 - b1) == 0):
        jList = [div_image[int(beta)][j1], div_image[int(beta)][j2]]
        p = np.interp(j, [j1, j2], jList)
        return p
    
    if ((j2 - j1) == 0):
        betaList = [div_image[b1][int(j)], div_image[b2][int(j)]]
        p = np.interp(beta, [b1, b2], betaList)
        return p

    result = (f11 * (b2 - beta) * (j2 - j) + f21 * (beta - b1) * (j2 - j) + f12 * (b2 - beta) * (j - j1) + f22 * (beta - b1) * (j - j1)) / ((b2 - b1) * (j2 - j1))
    
    return result

def contrast_stretching_2(image, intensity, pixelRange):
    # adjust so values are between 0 and 1
    if pixelRange > 1:
        image[image < 0] = 0.0
        image[image > 1] = 1.0
    if pixelRange == 1:
        image -= 40
        image[image < 0] = 0.0
        image[image > 255] = 255.0

    # adjust so values are between 0 and 255
    img_min = np.min(image)
    img_max = np.max(image)
    reconstruction_contrast_stretching = (image - img_min) / (img_max - img_min) * 255
    # Determine intensity of the contrast stretching
    reconstruction_contrast_stretching *= intensity
    return np.clip(reconstruction_contrast_stretching, 0, 255)

def contrast_stretching_1(image):
    # adjust so values are between 0 and 1
    image[image < 0] = 0.0
    image[image > 1] = 1.0

    # adjust so values are between 0 and 255
    img_min = np.min(image)
    img_max = np.max(image)
    reconstruction_contrast_stretching = (image - img_min) / (img_max - img_min) * 255
    # Determine intensity of the contrast stretching
    reconstruction_contrast_stretching *= 3
    return np.clip(reconstruction_contrast_stretching, 0, 255)

def is_360_sinogram(sinogram):
    num_projections = sinogram.shape[0]
    half_point = num_projections // 2
    similarity_threshold = 0.99  # margin of error is extremely low for a threshold of 0.99
    similar_count = 0

    for i in range(half_point):
        projection_1 = sinogram[i]
        # Flip the projection because the projection is done at the exact same angle but from the 'other' side of the circle
        projection_2 = np.flipud(sinogram[i + half_point])

        # Calculate the similarity between the two projections using correlation coefficient
        similarity = np.corrcoef(projection_1, projection_2)[0, 1]
        if similarity > similarity_threshold:
            similar_count += 1

    # If most pairs are similar, it should be a 360° sinogram
    return similar_count > half_point * 0.5

def is_vertically_stacked(sinogram):
    row_edge = sinogram[0, :]
    column_edge = sinogram[:, 0]
    avg_row = sum(row_edge) / len(row_edge)
    avg_column = sum(column_edge) / len(column_edge)
    print("AVG row value: {}\nAVG column value: {}".format(avg_row, avg_column))
    if avg_row > avg_column:
        return True
    else:
        return False

def CTSlice(sinogram, N, S):
    # Fourier transform the rows of the sinogram, move the DC component to the row's centre
    sinogram_fft_rows = fftshift(fft(ifftshift(sinogram)))

    # Polar coordinates of sinogram FFT-ed rows' samples in 2D FFT space (frequency domain)
    a = np.array([angle(i, N) for i in range(N)])
    r = np.arange(S) - S / 2
    r, a = np.meshgrid(r, a)
    r = r.flatten()
    a = a.flatten()

    # Convert polar coordinates to Cartesian coordinates
    srcx = (S / 2) + r * np.sin(a)
    srcy = (S / 2) + r * np.cos(a)

    # Create a Cartesian grid for the output image
    dstx, dsty = np.meshgrid(np.arange(S), np.arange(S))
    dstx = dstx.flatten()
    dsty = dsty.flatten()

    # Interpolate the 2D Fourier space grid from the transformed sinogram rows
    freq = scipy.interpolate.griddata(
        (srcy, srcx),
        sinogram_fft_rows.flatten(),
        (dstx, dsty),
        method="cubic",
        fill_value=0.0,
    ).reshape((S, S))

    # Inverse Fourier transform to reconstruct the image: take the real part to remove the imaginary part
    recon = np.real(fftshift(ifft2(ifftshift(freq))))

    return recon

def PrepareTarget(image):
    # Use the shape of the grayscale image for size
    size = image.shape[0]  # Size of target image, and resolution of Fourier space

    # If the image is larger than 512, scale it down: this is for testing purposes, rendering takes a long time with big resolutions
    if size > 512:
        size = 512

    # Create a meshgrid for the circular mask: only the portion of the image within the circle is retained, and the rest is set to zero
    x, y = np.meshgrid(np.arange(size) - size / 2, np.arange(size) - size / 2)

    # Create a circular mask
    mask = np.square(x) + np.square(y) <= np.square(size / 2 - 10)

    # Apply the mask to the resized image and fill the outside of the mask with zeros
    target = np.where(
        mask,
        resize(image, (size, size), anti_aliasing=True),  # Resize the image to the new size
        np.zeros((size, size)),  # Fill values outside the mask with zeros
    )

    return target, size

def angle(i, max_angle):
    return (math.pi * i) / max_angle

def func3_1(target, sinogram, intensity):
    # Prepare a target image: we use the Lotus image for this demo
    target, S = PrepareTarget(target)

    # Plotting the target image
    plt.subplot(2, 2, 1)
    plt.title("Target")
    plt.imshow(target)
    plt.gray()

    # Plotting the sinogram
    plt.subplot(2, 2, 2)
    plt.title("Sinogram of the target")
    plt.imshow(sinogram)
    plt.gray()

    # Check if we need to transpose the sinogram
    if (not is_vertically_stacked(sinogram)):
        sinogram = np.transpose(sinogram)

    # Check if it is a 180° or 360° sinogram and respond accordingly
    if is_360_sinogram(sinogram):
        print("It's a 360° sinogram")
        sinogram = sinogram[:180, :]
    else:
        print("It's a 180° sinogram")
    N = sinogram.shape[0]
    S = sinogram.shape[1]

    # Create the reconstruction using Fourier Slice Theorem
    reconstruction = CTSlice(sinogram, N, S)
    plt.subplot(2, 2, 3)
    plt.title("CTSlice: Reconstruction of the sinogram")
    plt.imshow(reconstruction)
    plt.gray()

    # Add contrast stretching to the result
    reconstruction = contrast_stretching_2(reconstruction, intensity, 255)

    # Plot the reconstruction
    plt.subplot(2, 2, 4)
    plt.title("CTSlice: Reconstruction of the sinogram with contrast stretching")
    plt.imshow(reconstruction)
    plt.gray()

    plt.show()

    return reconstruction

def func3_2(target, sinogram, intensity, ram_lak_filter=False):
    target, S = PrepareTarget(target)

    # Plotting the image
    plt.subplot(3, 2, 1)
    plt.title("Target")
    plt.imshow(target)  # Set aspect ratio to 'equal'
    plt.gray()

    # lotus is transposed
    # Check if we need to transpose the sinogram
    if (not is_vertically_stacked(sinogram)):
        sinogram = np.transpose(sinogram)

    # half the input because of the way the sinogram is created
    # Check if it is a 180° or 360° sinogram and respond accordingly
    if is_360_sinogram(sinogram):
        print("It's a 360° sinogram")
        sinogram = sinogram[:180, :]
    else:
        print("It's a 180° sinogram")
    N = sinogram.shape[0]
    S = sinogram.shape[1]

    if not ram_lak_filter:
        #Ramp filter
        ramp = np.linspace(0, 1, S)
    else:
        #ram lak filter
        temp = np.linspace(1, 0, S)
        ramp = np.sinc(temp)

    plt.subplot(3, 2, 2)
    plt.title("Sinogram")
    plt.imshow(sinogram)
    plt.gray()

    # execute Filtered backprojection and then add it to the subplots
    reconstruction_FB = FB(sinogram, N, S, ramp)
    reconstruction_FB_contrast_stretched = contrast_stretching_2(reconstruction_FB, intensity, 255)

    plt.subplot(3, 2, 4)
    plt.title("Reconstruction FB")
    plt.imshow(reconstruction_FB)
    plt.gray()

    plt.subplot(3, 2, 5)
    plt.title("Reconstruction FB Contrast Stretching")
    plt.imshow(reconstruction_FB_contrast_stretched)
    plt.gray()

    plt.show()

def func3_3(div_sino, sinogram):
    real_div = div_sino
    if (not is_vertically_stacked(real_div)):
        real_div = cv2.rotate(real_div, cv2.ROTATE_90_COUNTERCLOCKWISE)
    para_sinogram = Rebinning(real_div)

    # Plot the rebinned image
    plt.subplot(1, 3, 1)
    plt.title("Divergent sinogram")
    plt.imshow(real_div)
    plt.gray()

    plt.subplot(1, 3, 2)
    plt.title("Real parallel sinogram")
    plt.imshow(sinogram)
    plt.gray()

    plt.subplot(1, 3, 3)
    plt.title("Rebinned sinogram")
    plt.imshow(para_sinogram)
    plt.gray()

    plt.show()

def CreateSino(target, max_angle):
    # Project the sinogram (ie calculate Radon transform)
    sinogram = np.array(
        [np.sum(scipy.ndimage.rotate(target, np.rad2deg(angle(i, max_angle)), order=3, reshape=False, mode="constant", cval=0.0, ),
                axis=0, ) for i in range(max_angle)]
    )
    return sinogram

def func3_4(image, sinogram):
    # Plotting the target image
    plt.subplot(2, 2, 1)
    plt.title("Target")
    plt.imshow(image)
    plt.gray()

    # Plotting the sinogram
    plt.subplot(2, 2, 2)
    plt.title("Real sinogram of the target")
    plt.imshow(sinogram)
    plt.gray()

    # Creating the sinogram
    sinogram = CreateSinogram(image, 180, 180)
    plt.subplot(2, 2, 3)
    plt.title("Created sinogram of the target")
    plt.imshow(sinogram)
    plt.gray()

    # Create CTSlice from the created sinogram
    # Create the reconstruction using Fourier Slice Theorem
    N = sinogram.shape[0]
    S = sinogram.shape[1]
    reconstruction = CTSlice(sinogram, N, S)
    # Plot the reconstruction
    plt.subplot(2, 2, 4)
    plt.title("CTSlice: Reconstruction of the sinogram")
    plt.imshow(reconstruction)
    plt.gray()

    plt.show()

def func3_5_1(sinogram):
    # Check if we need to transpose the sinogram
    if (not is_vertically_stacked(sinogram)):
        sinogram = np.transpose(sinogram)

    if is_360_sinogram(sinogram):
        print("It's a 360° sinogram")
    else:
        print("It's a 180° sinogram")


if __name__ == '__main__':
    main()
