# Full CT image reconstruction pipeline using the Fourier Slice Theorem and Filtered Backprojection.
# Accepts a sinogram and optional ground-truth image via CLI arguments, automatically detects
# sinogram orientation and angular coverage, reconstructs the image, and displays the results.
#
# Created: 2024-10-28
# Author: Devon Vanaenrode

# --- Imports ---
import argparse
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import scipy.interpolate
import scipy.ndimage
from skimage.transform import rotate, resize
from scipy.fftpack import fft, ifft, ifft2, fftshift, ifftshift


def load_image(path: str, rotate_img: bool = False) -> np.ndarray:
    """Loads a grayscale image from the given path."""
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {path}")
    
    if rotate_img:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def get_angle_radians(i: int, max_angle: int) -> float:
    """Calculates the angle in radians for a given projection step."""
    return (math.pi * i) / max_angle


def prepare_target(image_path: str) -> tuple[np.ndarray, int]:
    """
    Prepares the target image by resizing to a maximum of 512x512
    and applying a circular mask to isolate the region of interest.
    """
    image = load_image(image_path)
    size = min(image.shape[0], 512)

    # Create a circular mask
    x, y = np.meshgrid(np.arange(size) - size / 2, np.arange(size) - size / 2)
    mask = np.square(x) + np.square(y) <= np.square(size / 2 - 10)

    # Apply mask and pad the outside with zeros
    target = np.where(
        mask,
        resize(image, (size, size), anti_aliasing=True),
        np.zeros((size, size)),
    )
    return target, size


def create_sinogram(target: np.ndarray, max_angle: int) -> np.ndarray:
    """Calculates the Radon transform (sinogram) of the target image."""
    sinogram = np.array([
        np.sum(scipy.ndimage.rotate(
            target, np.rad2deg(get_angle_radians(i, max_angle)), 
            order=3, reshape=False, mode="constant", cval=0.0
        ), axis=0) for i in range(max_angle)
    ])
    return sinogram


def back_project(sinogram: np.ndarray, angular_range: float = 180.0) -> np.ndarray:
    """Performs standard backprojection on a sinogram."""
    laminogram = np.zeros((sinogram.shape[1], sinogram.shape[1]))
    d_theta = angular_range / sinogram.shape[0]

    for i in range(sinogram.shape[0]):
        temp = np.tile(sinogram[i], (sinogram.shape[1], 1))
        temp = rotate(np.absolute(temp), d_theta * i)
        laminogram += temp
    return laminogram


def ramp_filter(ffts: np.ndarray) -> np.ndarray:
    """Applies a ramp filter to a 2D array of 1D FFTs along the rows."""
    ramp = np.floor(np.arange(0.5, ffts.shape[1] // 2 + 0.1, 0.5))
    return ffts * ramp


def filtered_backprojection(sinogram: np.ndarray) -> np.ndarray:
    """Reconstructs an image using Filtered Backprojection (FBP)."""
    frequency_domain_sinogram = fft(sinogram, axis=1)
    filtered_freq_domain_sinogram = ramp_filter(frequency_domain_sinogram)
    filtered_spatial_domain_sinogram = ifft(filtered_freq_domain_sinogram, axis=1)
    
    reconstructed_image = back_project(filtered_spatial_domain_sinogram)

    # Apply Hamming window
    window_1d = np.abs(np.hamming(reconstructed_image.shape[0]))
    window_2d = np.sqrt(np.outer(window_1d, window_1d))
    
    return reconstructed_image * window_2d


def fourier_slice_reconstruction(sinogram: np.ndarray, num_projections: int, num_samples: int) -> np.ndarray:
    """
    Reconstructs the image using the Fourier Slice Theorem.
    Transforms sinogram rows into 2D Fourier space and interpolates to Cartesian coordinates.
    """
    # Fourier transform the rows and center the DC component
    sinogram_fft_rows = fftshift(fft(ifftshift(sinogram)))

    # Polar coordinates of samples in 2D frequency domain
    a = np.array([get_angle_radians(i, num_projections) for i in range(num_projections)])
    r = np.arange(num_samples) - num_samples / 2
    r, a = np.meshgrid(r, a)
    r, a = r.flatten(), a.flatten()

    # Convert polar to Cartesian coordinates
    src_x = (num_samples / 2) + r * np.sin(a)
    src_y = (num_samples / 2) + r * np.cos(a)

    dst_x, dst_y = np.meshgrid(np.arange(num_samples), np.arange(num_samples))

    # Interpolate to Cartesian grid
    freq = scipy.interpolate.griddata(
        (src_y, src_x),
        sinogram_fft_rows.flatten(),
        (dst_x.flatten(), dst_y.flatten()),
        method="cubic",
        fill_value=0.0,
    ).reshape((num_samples, num_samples))

    # Inverse 2D FFT to reconstruct the image (take real part)
    recon = np.real(fftshift(ifft2(ifftshift(freq))))
    return recon


def is_360_sinogram(sinogram: np.ndarray) -> bool:
    """Determines if the provided sinogram covers 180 or 360 degrees using correlation."""
    num_projections = sinogram.shape[0]
    half_point = num_projections // 2
    similarity_threshold = 0.99  
    similar_count = 0

    for i in range(half_point):
        projection_1 = sinogram[i]
        # Flip the projection from the opposite side of the circle
        projection_2 = np.flipud(sinogram[i + half_point])

        similarity = np.corrcoef(projection_1, projection_2)[0, 1]
        if similarity > similarity_threshold:
            similar_count += 1

    return similar_count > half_point * 0.5


def contrast_stretching(image: np.ndarray) -> np.ndarray:
    """Normalizes and stretches the contrast of the reconstructed image."""
    image = np.clip(image, 0.0, 1.0)
    img_min, img_max = np.min(image), np.max(image)
    
    if img_min == img_max:
        return np.zeros_like(image)

    stretched = (image - img_min) / (img_max - img_min) * 255.0
    stretched *= 3  # Boost intensity
    return np.clip(stretched, 0, 255)


def main():
    parser = argparse.ArgumentParser(description="CT Image Reconstruction using Fourier Slice Theorem")
    parser.add_argument("--target", type=str, default="Samples/lotus.png", help="Path to the target ground-truth image")
    parser.add_argument("--sinogram", type=str, default="Samples/lotus_parallel.png", help="Path to the input sinogram image")
    args = parser.parse_args()

    print("[*] Preparing target image...")
    try:
        target, num_samples = prepare_target(args.target)
    except FileNotFoundError:
        print(f"[!] Warning: Target image '{args.target}' not found. Ground truth will not be displayed.")
        target = None

    print(f"[*] Loading sinogram from {args.sinogram}...")
    sinogram = load_image(args.sinogram)
    sinogram = np.transpose(sinogram)

    if is_360_sinogram(sinogram):
        print("[+] Detected a 360° sinogram. Truncating to 180° for reconstruction.")
        sinogram = sinogram[:180, :]
    else:
        print("[+] Detected a 180° sinogram.")

    num_projections, num_samples = sinogram.shape

    print(f"[*] Reconstructing image (Projections: {num_projections}, Samples: {num_samples})...")
    reconstruction = fourier_slice_reconstruction(sinogram, num_projections, num_samples)
    reconstruction_stretched = contrast_stretching(reconstruction)

    print("[*] Plotting results...")
    # Grouping the visualizations into a single, clean dashboard
    fig, axes = plt.subplots(1, 4 if target is not None else 3, figsize=(16, 4))
    
    plot_idx = 0
    if target is not None:
        axes[plot_idx].imshow(target, cmap="gray")
        axes[plot_idx].set_title("Target (Ground Truth)")
        axes[plot_idx].axis('off')
        plot_idx += 1
        
    axes[plot_idx].imshow(sinogram, cmap="gray")
    axes[plot_idx].set_title("Sinogram")
    plot_idx += 1
    
    axes[plot_idx].imshow(reconstruction, cmap="gray")
    axes[plot_idx].set_title("Reconstruction (Raw)")
    axes[plot_idx].axis('off')
    plot_idx += 1
    
    axes[plot_idx].imshow(reconstruction_stretched, cmap="gray")
    axes[plot_idx].set_title("Reconstruction (Enhanced)")
    axes[plot_idx].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()