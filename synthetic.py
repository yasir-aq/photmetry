import numpy as np
from astropy.io import fits
from pathlib import Path

# Define the directory containing the FITS files
flat_directory = Path('/Users/yasir/Documents/20220407')

# List all FITS files in the directory
fits_files = list(flat_directory.glob('*.fits'))

# Initialize a list to store image data
images = []

# Load each FITS file and append its data to the list
for fits_file in fits_files:
    with fits.open(fits_file) as hdul:
        image_data = hdul[0].data
        images.append(image_data)

# Convert list of images to a 3D numpy array (n_images, height, width)
image_stack = np.array(images)

# Ensure that there are images to process
if image_stack.size == 0:
    raise ValueError("No FITS images found in the directory.")
    
# Ensure all images have the same shape
if len(set(image.shape for image in images)) != 1:
    raise ValueError("All FITS images must have the same shape.")
    
# Create the synthetic flat field by taking the minimum pixel value across all images
flat_field = np.min(image_stack, axis=0)

# Normalize the synthetic flat field to the range [0, 1]
flat_field_min = np.min(flat_field)
flat_field_max = np.max(flat_field)

# Avoid division by zero if all values are the same
if flat_field_max != flat_field_min:
    flat_field_normalized = (flat_field - flat_field_min) / (flat_field_max - flat_field_min)
else:
    flat_field_normalized = np.zeros_like(flat_field)  # If all values are the same, normalize to zero

# Define the output file path
output_file = flat_directory / 'synthetic_flat_normalized.fits'

# Save the normalized synthetic flat field image to a FITS file
hdu = fits.PrimaryHDU(flat_field_normalized)
hdul = fits.HDUList([hdu])
hdul.writeto(output_file, overwrite=True)

print(f"Synthetic normalized flat field image saved to {output_file}")
