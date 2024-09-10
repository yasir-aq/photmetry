import numpy as np
from astropy.io import fits
import os

def load_fits(file_path):
    """ Load a FITS file and return its data """
    with fits.open(file_path) as hdul:
        data = hdul[0].data.astype(np.float32)
    return data

def save_fits(data, file_path, header=None):
    """ Save data to a FITS file """
    hdu = fits.PrimaryHDU(data=data, header=header)
    hdu.writeto(file_path, overwrite=True)

def create_master_bias(bias_files):
    """ Create a master bias frame by taking the median of all bias frames """
    bias_stack = np.array([load_fits(file) for file in bias_files])
    master_bias = np.median(bias_stack, axis=0)
    return master_bias

def create_master_flat(flat_files, master_bias):
    """ Create a master flat frame by taking the median of all bias-corrected flat frames """
    flat_stack = []
    for file in flat_files:
        flat_data = load_fits(file)
        bias_corrected_flat = flat_data - master_bias  # Subtract the master bias from each flat frame
        flat_stack.append(bias_corrected_flat)
    flat_stack = np.array(flat_stack)
    master_flat = np.median(flat_stack, axis=0)
    master_flat /= np.median(master_flat)  # Normalize the master flat to have a median value of 1
    return master_flat

def destripe_image(image):
    """ Apply destriping by subtracting row medians """
    for i in range(image.shape[0]):
        row_median = np.median(image[i, :])
        image[i, :] -= row_median
    return image

def fix_dead_pixels(image, threshold=1.5):
    """ 
    Fix isolated dead pixels that are brighter than 1.5 times their neighboring pixels.
    The pixel is replaced by the average of the neighboring three pixels.
    """
    # Create a copy of the image to avoid modifying the original during the loop
    fixed_image = image.copy()
    
    # Define offsets for neighboring pixels
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            # Get the value of the current pixel
            pixel_value = image[i, j]
            
            # Get the values of the neighboring pixels
            neighbors = [image[i + di, j + dj] for di, dj in offsets]
            
            # Calculate the average of the neighboring pixels
            neighbor_avg = np.mean(neighbors)
            
            # Check if the current pixel is an isolated dead pixel
            if pixel_value > threshold * neighbor_avg:
                # Replace the dead pixel with the average of the neighboring pixels
                fixed_image[i, j] = neighbor_avg
    
    return fixed_image

def process_image(science_image, master_bias, master_flat):
    """ Apply bias subtraction, flat-field correction, destriping, and dead pixel fixing """
    # Bias correction
    corrected_image = science_image - master_bias
    
    # Flat field correction
    corrected_image /= master_flat
    
    # Destriping
    destriped_image = destripe_image(corrected_image)
    
    # Fix dead pixels
    final_image = fix_dead_pixels(destriped_image)
    
    return final_image

def process_directory(science_dir, master_bias, master_flat, output_dir):
    """ Process all science images in a directory """
    # Process each science image in the directory
    for file_name in os.listdir(science_dir):
        if file_name.endswith('.fits'):
            science_image_path = os.path.join(science_dir, file_name)
            science_image = load_fits(science_image_path)
            
            # Process the image
            processed_image = process_image(science_image, master_bias, master_flat)
            
            # Save the processed image
            output_path = os.path.join(output_dir, file_name)
            save_fits(processed_image, output_path)
            print(f"Processed and saved: {output_path}")

# Define paths for bias and flat files
bias_dir = '/Users/yasir/Documents/20240312/bias'
flat_dir = '/Users/yasir/Documents/20240312/flat'
science_dir = '/Users/yasir/Documents/20240312/raw'
output_dir = '/Users/yasir/Documents/20240312/processed0'

# Create list of all bias and flat FITS files
bias_files = [os.path.join(bias_dir, f) for f in os.listdir(bias_dir) if f.endswith('.fits')]
flat_files = [os.path.join(flat_dir, f) for f in os.listdir(flat_dir) if f.endswith('.fits')]

# Create master bias and master flat
master_bias = create_master_bias(bias_files)
save_fits(master_bias, '/Users/yasir/Documents/20240312/bias/master_bias.fits')

master_flat = create_master_flat(flat_files, master_bias)
save_fits(master_flat, '/Users/yasir/Documents/20240312/flat/master_flat.fits')

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process all science images
process_directory(science_dir, master_bias, master_flat, output_dir)
