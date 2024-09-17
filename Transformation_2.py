import numpy as np
from plantcv import plantcv as pcv
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

from pilkit.lib import Image
from pilkit.processors import TrimBorderColor

import os

#plant cv Documentation

# https://github.com/danforthcenter/plantcv/tree/main/docs


# Set global debug behavior to None (default), "print" (to file),
# or "plot" (Jupyter Notebooks or X11)
pcv.params.debug = "print"



# def read_image(image_path):
#     #Read image
#     img = Image.open(image_path)
#
#     #Transform to numpy array
#     np_img = np.asarray(img)
#     # img_name, img_extension = os.path.splitext(image_path)


def transform_image(image_path):

    # I can read image also like this:
    # Read image
    img = Image.open(image_path)

    # Transform to numpy array
    np_img = np.asarray(img)

    # #But this is less code
    # # read in image
    # np_img, path, img_filename = pcv.readimage(filename=image_path, mode="native")
    # print(path)
    # print(img_filename)

    # Optionally, set a sample label name
    pcv.params.sample_label = "plant"

    # 1 Original
    # pcv.print_image(np_img, "0_original.png")

    ####1. Methods of Isolating Target Objects
    #####Object Segmentation Approaches
    #####Thresholding method
    #2_1 Thresholded image from gray
    # image converted from RGB to gray.
    gray_img = pcv.rgb2gray(rgb_img=np_img)
    # Create binary image from a gray image based on threshold values,
    # targeting light objects in the image.
    # Perform thresholding to generate a binary image
    threshold_light = pcv.threshold.binary(gray_img=gray_img, threshold=120, object_type='dark')
    #delete small parts
    bsa_fill1=pcv.fill(bin_img = threshold_light, size = 15)

    # bsa_fill1 = pcv.dilate(gray_img=bsa_fill1, ksize=1.5, i = 2)

    #closing small pixels
    filled_mask1 = pcv.closing(gray_img = bsa_fill1)
    # pcv.print_image(filled_mask1, "1_binary.png")

    #####Noise Reduction
    #2 Gaussian blur DONE
    # Apply gaussian blur to a binary image that has been previously thresholded.
    gaussian_img = pcv.gaussian_blur(img=filled_mask1, ksize=(3, 3), sigma_x=0, sigma_y=None)
    # img_flip.save(f"{img_name}_Flip{img_extension}")
    # pcv.print_image(gaussian_img, "2_gaussian_image.png")

    # #3 Mask
    # # Apply binary 'white' mask over an image.
    masked_image = pcv.apply_mask(img=np_img, mask=gaussian_img, mask_color='white')
    # pcv.print_image(masked_image, "3_mask.png")

    #####Region of Interest
    #4 Roi objects (Region of interest to mask)
    # ROI filter allows the user to define if objects partially inside ROI are included or if objects are cut to ROI.
    # Make a grid of ROIs done
    center_x = int(np_img.shape[0]/2)
    print(center_x)
    center_y = int(np_img.shape[1] / 2)
    print(center_y)
    radius = int((center_x + center_y )/2)
    print(radius)
    roi = pcv.roi.circle(img=np_img, x=center_x, y=center_y, r=75)
    # filtered_mask = pcv.roi.filter(mask=bsa_fill1, roi=roi, roi_type='partial')
    # pcv.print_image(filtered_mask, "5_roi.png")

    ####2. Object Analysis in PlantCV
    # #5 Analyze object - Analyze plant shape
    # Characterize object shapes
    shape_image = pcv.analyze.size(img=np_img, labeled_mask=bsa_fill1, n_labels=1)
    # pcv.print_image(shape_image, "6.png")

    #6 Pseudolandmarks
    device = 1
    # Identify a set of land mark points
    # Results in set of point values that may indicate tip points
    top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(img=np_img, mask=bsa_fill1)

    # 7 Histogram Analyze Color DONE
    # Examine signal distribution within an image
    # prints out an image histogram of signal within image
    hist_figure, hist_data = pcv.visualize.histogram(img=np_img, mask=bsa_fill1, hist_data=True)

    # # Green-Magenta ('a') channel is output
    # a_channel = pcv.rgb2gray_lab(rgb_img=np_img, channel='a')
    #
    # hist_figure_a, hist_data_a = pcv.visualize.histogram(img=a_channel, mask=bsa_fill1, hist_data=True)

    # analysis_images = pcv.analyze.color(np_img, bsa_fill1)
    # pcv.print_image(analysis_images, "8_analysis_images.png")
    # pcv.outputs.save_results(filename="results")

if __name__ == "__main__":
    directory = "/Users/air/Documents/ecole/leaffliction/images/Apple_Black_rot/image (1).JPG"
    # Apply transformation
    transform_image(directory)