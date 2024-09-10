#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 10:40:27 2024

@author: yasir
"""

import numpy as np
from astropy.io import fits
from pathlib import Path
from scipy.ndimage import median_filter
import ccdproc as ccdp
from astropy.nddata import CCDData

# Directories
science_directory = Path('/Users/yasir/Documents/20240312/')
masked_directory = Path('/Users/yasir/Documents/20240312/masked/')
calibrated_directory = Path('/Users/yasir/Documents/20240312/calibrated/')
masked_directory.mkdir(parents=True, exist_ok=True)
calibrated_directory.mkdir(parents=True, exist_ok=True)

# Paths to master flat
master_flat_path = Path('/Users/yasir/Documents/20240312/flat/synthetic_flat_corrected.fits')
master_flat = CCDData.read(master_flat_path, unit='adu')

# Parameters for cosmic ray detection and row correction
filter_size = 5  # Size of the median filter
row_threshold_factor = 5  # Factor for row-wise correction threshold

def mask_cosmic_rays(image, filter_size):
    filtered_image = median_filter(image, size=filter_size)
    threshold = row_threshold_factor * np.median(np.abs(image - filtered_image))
    cosmic_ray_mask = np.abs(image - filtered_image) > threshold
    corrected_image = np.where(cosmic_ray_mask, filtered_image, image)
    return corrected_image

def correct_row_issues(image):
    row_means = np.mean(image, axis=1, keepdims=True)
    row_corrected_image = image / row_means
    row_corrected_image[np.isnan(row_corrected_image)] = 0
    row_corrected_image[np.isinf(row_corrected_image)] = 0
    return row_corrected_image

# Process each raw science image
for science_file in science_directory.glob('*.fits'):
    # Read raw science image and header
    with fits.open(science_file) as hdul:
        science_image = hdul[0].data
        header = hdul[0].header

    # Mask cosmic rays
    masked_image = mask_cosmic_rays(science_image, filter_size)

    # Correct row-wise light/dark issues
    row_corrected_image = correct_row_issues(masked_image)

    # Save the processed science image with the original header
    masked_image_path = masked_directory / science_file.name
    hdu = fits.PrimaryHDU(data=row_corrected_image, header=header)
    hdu.writeto(masked_image_path, overwrite=True)

    # Apply flat field correction
    science_image_ccd = CCDData.read(masked_image_path, unit='adu')
    calibrated_image = ccdp.flat_correct(science_image_ccd, master_flat)

    # Save the calibrated science image with the original header
    calibrated_image_path = calibrated_directory / science_file.name
    calibrated_image.write(calibrated_image_path, overwrite=True)

    print(f"Processed and saved: {calibrated_image_path}")
