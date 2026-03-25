import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq

def ct_reconstruction_fourier_slice_theorem(sinogram):
    num_projections, num_angles = sinogram.shape

    # Step 1: Compute 1D Fourier Transform along the projections
    f_sinogram = fft(sinogram, axis=0)

    # Step 2: Create frequency axis
    freq_axis = fftfreq(num_projections)

    # Step 3: Initialize the reconstructed image
    reconstruction = np.zeros((num_projections, num_projections))

    # Step 4: Apply the Fourier Slice Theorem
    for angle in range(num_angles):
        projection_angle = angle * np.pi / num_angles
        filtered_projection = f_sinogram[:, angle] * np.exp(-1j * freq_axis * projection_angle)
        slice_profile = ifft(filtered_projection)

        # Interpolate and add to the reconstruction
        reconstruction[:, angle] += np.real(slice_profile)

    return reconstruction

# Example usage
# Generate a simple phantom image
size = 256
phantom = np.zeros((size, size))
phantom[size//4:3*size//4, size//4:3*size//4] = 1.0

# Generate sinogram from the phantom image
angles = 180
sinogram = np.zeros((size, angles))
for angle in range(angles):
    rotated_phantom = np.rot90(phantom, angle)
    sinogram[:, angle] = np.sum(rotated_phantom, axis=0)

# Perform CT reconstruction using Fourier Slice Theorem
reconstruction = ct_reconstruction_fourier_slice_theorem(sinogram)

# Display the results
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(phantom, cmap='gray')
plt.title('Original Phantom')

plt.subplot(1, 3, 2)
plt.imshow(sinogram, cmap='gray', aspect='auto')
plt.title('Sinogram')

plt.subplot(1, 3, 3)
plt.imshow(reconstruction, cmap='gray')
plt.title('Reconstructed Image')

plt.show()
