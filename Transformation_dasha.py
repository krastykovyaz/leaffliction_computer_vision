#!/usr/bin/env python3

from plantcv import plantcv as pcv
from pilkit.lib import Image
import logging
from argparse import ArgumentParser
from argparse import ArgumentTypeError
import matplotlib
import pandas as pd
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
# from pilkit.processors import TrimBorderColor
# from PIL import Image, ImageOps, ImageEnhance, ImageFilter


def gaussian_blur(image):
    return cv2.GaussianBlur(image, (15, 15), 0)


def create_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    return mask


def analyze_object(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray,
                                   cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    return image


def display_image_transformations(image_path):
    image = cv2.imread(image_path)

    # Apply transformations
    blur = gaussian_blur(image)
    mask = create_mask(image)
    analyzed = analyze_object(image.copy())

    # Display results
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original")

    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB))
    plt.title("Gaussian Blur")

    plt.subplot(2, 2, 3)
    plt.imshow(mask, cmap='gray')
    plt.title("Mask")

    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(analyzed, cv2.COLOR_BGR2RGB))
    plt.title("Analyzed Object")

    plt.show()

def by_chanels(chans, colors, labels):
    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        plt.plot(hist, color=color, label=labels)
        plt.xlim([0, 256])

def plot_color_histogram(image_path):
    image = cv2.imread(image_path)

    # Transform to numpy array
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    chans = cv2.split(image)
    chans_hsv = cv2.split(hsv_img)
    chans_lab = cv2.split(lab_img)

    plt.figure()
    plt.title("Color Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")

    by_chanels(chans[0], "b", "blue")
    by_chanels(chans[1], "g", "green")
    by_chanels(chans[2], "r", "red")
    by_chanels(chans_hsv[0], "m", "hue")
    by_chanels(chans_hsv[1], "c", "saturation")
    by_chanels(chans_hsv[2], "y", "value")
    by_chanels(chans_lab[0], "b", "lightness")
    by_chanels(chans_lab[1], "c", "green_red")
    by_chanels(chans_lab[2], "y", "blue_yellow")
    plt.legend()
    plt.show()

# Example usage
# plot_color_histogram("images/Apple/apple_healthy/image (1).JPG")


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise ArgumentTypeError(f"readable_dir:{path} is not a valid path")


def file_img(img):
    # hsv_img = matplotlib.colors.rgb_to_hsv(np_img)
    # check if it is a jpg file
    if os.path.abspath(img).lower().endswith(".jpg"):
        return img
    else:
        raise ArgumentTypeError(f"{img} is not a valid image")


parser = ArgumentParser(description=__doc__)
#

parser.add_argument("img_src", help="Path to the input image", type=file_img)

parser.add_argument(
    "-src",
    # dest="first_string",
    help="the source directory of images",
    type=dir_path
    # required=True,
)

parser.add_argument(
    "-dst",
    # dest="second_string",
    help="destinagtion directory for transformed images",
    type=dir_path
    # required=True,
)

parser.add_argument("-gaussian", help="apply gaussian blur",
                    action="store_true")
parser.add_argument("-mask", help="apply mask",
                    action="store_true")
parser.add_argument("-roi", help="apply region of interest",
                    action="store_true")
parser.add_argument("-analyze", help="apply Analyze object",
                    action="store_true")
parser.add_argument("-pseudolandmarks", help="apply pseudolandmarks",
                    action="store_true")
parser.add_argument("-histogram", help="apply color histogram",
                    action="store_true")


# input_parameters, unknown_input_parameters = parser.parse_known_args()
#
# # Set CLI argument variables.
# first_arg = input_parameters.first_string
# second_arg = input_parameters.second_string
# third_arg = input_parameters.third_string
#
# print("-src: {}\n"
#       "-dest: {}\n"
#       "-i: {}".format(first_arg, second_arg, third_arg))
#


# plant cv Documentation
# https://github.com/danforthcenter/plantcv/tree/main/docs


# Set global debug behavior to None (default), "print" (to file),
# or "plot" (Jupyter Notebooks or X11)
pcv.params.debug = "None"

def pcv_histogram(image_path, np_img, my_mask):
    # 7 Histogram Analyze Color DONE
    # Examine signal distribution within an image
    # prints out an image histogram of signal within image
    logger.info("saving histogram")
    hist_figure, hist_data = pcv.visualize.histogram(img=np_img,
                                                     title="histogram",
                                                     mask=my_mask,
                                                     hist_data=True)
    image = cv2.imread(image_path)
    # hsv_img = matplotlib.colors.rgb_to_hsv(np_img)
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_figure_hsv, hist_data_hsv = pcv.visualize.histogram(img=hsv_img, mask=my_mask, hist_data=True)
    pcv.print_image(hist_figure, filename = "histogram.png")
    pass

def pcv_roi(np_img, bsa_fill1, name):
    new_name = name + "roi.png"
    # ####Region of Interest
    # 4 Roi objects (Region of interest to mask)
    # ROI filter allows the user to define if objects partially
    # inside ROI are included or if objects are cut to ROI.
    # Make a grid of ROIs done
    center_x = int(np_img.shape[0] / 2)
    print(center_x)
    center_y = int(np_img.shape[1] / 2)
    print(center_y)
    radius = int((center_x + center_y) / 2)
    print(radius)
    roi = pcv.roi.circle(img=np_img, x=center_x, y=center_y, r=75)
    print("roi", type(roi))
    print("my roi", roi.__str__())

    # Make a new filtered mask that only keeps the plant in your ROI and not objects outside of the ROI
    # We have set to partial here so that if a leaf extends outside of your ROI it will still be selected. Switch to "cutto" if you have other plants that are getting selected on accident

    # Inputs:
    #    mask            = the clean mask you made above
    #    roi            = the region of interest you specified above
    #    roi_type       = 'partial' (default, for partially inside the ROI), 'cutto', or
    #                     'largest' (keep only largest contour)
    filtered_mask = pcv.roi.filter(mask=bsa_fill1,
                                   roi=roi, roi_type='partial')
    pcv.print_image(filtered_mask, new_name)
    return center_x, center_y, radius

def transform_image_1(image_path):

    # I can read image also like this:
    # Read image
    img = Image.open(image_path)

    # Transform to numpy array
    np_img = np.asarray(img)

    print(image_path)
    plot_color_histogram(image_path)
    pcv_histogram(image_path, np_img, None)
    pass

def pcv_analyze_object(img, mask, name):
    # Characterize object shapes
    newname = name + "_analyze_object.png"
    shape_img = pcv.analyze.size(img=img,
                                   labeled_mask=mask,
                                   n_labels=1)
    logger.info("saving analyze object")
    pcv.print_image(shape_img, newname)
    return shape_img
def pcv_pseudolanmarks(np_img, mask, name):
    device = 1
    # Identify a set of land mark points
    # Results in set of point values that may indicate tip points
    top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(img=np_img, mask=mask)
    print("np", type(np_img))
    print("top", type(top))
    print("bottom", type(bottom))
    print("center", type(center_v))
    pseudo_img = pcv.apply_mask(img=np_img,
                                    mask=top,
                                    mask_color='black')
    return top, bottom, center_v


def transform_image(image_path, gaussian=False, mask=False, roi=False, analyze=False, pseudolandmarks=False, histogram=False):
    print(image_path)
    # I can read image also like this:
    img = Image.open(image_path)
    # Transform to numpy array
    np_img = np.asarray(img)

    # #But this is less code of reading image
    # np_img, path, img_filename = pcv.readimage(filename=image_path, mode="native")

    # Update params related to plotting so we can see better
    pcv.params.text_size = 5
    pcv.params.text_thickness = 5
    # Look at the colorspace - which of these looks the best for masking?
    # Which channel makes the plant look most distinct from the background?
    colorspace_img = pcv.visualize.colorspaces(rgb_img=np_img, original_img=False)
    pcv.print_image(colorspace_img, "colorspace_image.png")


    # # Optionally, set a sample label name
    # pcv.params.sample_label = "plant"

    # 1 Original
    # pcv.print_image(np_img, "0_original.png")

    # ###1. Methods of Isolating Target Objects
    # ####Object Segmentation Approaches
    # Thresholding method
    # 2_1 Thresholded image from gray
    # image converted from RGB to gray.
    gray_img = pcv.rgb2gray(rgb_img=np_img)
    # Create binary image from a gray image based on threshold values,
    threshold_light = pcv.threshold.binary(gray_img=gray_img,
                                           threshold=115,
                                           object_type='dark')
    # delete small parts of pixels
    bsa_fill1 = pcv.fill(bin_img=threshold_light,
                         size=25)

    # bsa_fill1 = pcv.dilate(gray_img=bsa_fill1, ksize=1.5, i = 2)

    # closing small pixels
    filled_mask1 = pcv.closing(gray_img=bsa_fill1)
    # pcv.print_image(filled_mask1, "1_binary.png")

    # Noise Reduction
    # 2 Gaussian blur
    # Apply gaussian blur to a binary image
    gaussian_img = pcv.gaussian_blur(img=filled_mask1,
                                     ksize=(3, 3),
                                     sigma_x=0,
                                     sigma_y=None)
    # img_flip.save(f"{img_name}_Flip{img_extension}")
    if gaussian:
        pcv.print_image(gaussian_img, "2_gaussian_image.png")

    print("mask", type(filled_mask1))
    # #3 Mask - Apply binary 'white' mask over an image.
    masked_img = pcv.apply_mask(img=np_img,
                                  mask=gaussian_img,
                                  mask_color='white')
    # if mask:
    pcv.print_image(masked_img, "masked_image.png")
    img_name = "plant"

    # if roi:
    center_x, center_y, radius = pcv_roi(np_img, bsa_fill1, img_name)

    # if analyze:
    shape_img = pcv_analyze_object(np_img, bsa_fill1, img_name)

    # if pseudolandmarks:
    top, bottom, center_v = pcv_pseudolanmarks(np_img, bsa_fill1, img_name)

    # Display results
    plt.figure(figsize=(10, 8))
    # plt.title(figure_title, y=1.08)

    plt.subplot(3, 2, 1)
    plt.imshow(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
    plt.title("1: Original")

    print(type(np_img))
    plt.subplot(3, 2, 2)
    plt.imshow(cv2.cvtColor(gaussian_img, cv2.COLOR_BGR2RGB))
    plt.title("2: Gaussian Blur")

    plt.subplot(3, 2, 3)
    plt.imshow(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
    plt.title("3: Analyzed Object")

    fig = plt.subplot(3, 2, 4)
    plt.imshow(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
    circle = plt.Circle((center_x, center_y), radius*0.8, alpha = 1, fill = None, linewidth=5,  facecolor=None,  linestyle='-', edgecolor = "blue")
    plt.scatter(center_x, center_y, 10, facecolors='none', edgecolors='blue')
    fig.add_patch(circle)
    # x = center_x, y = center_y, r = 75
    plt.title("4: ROI objects")

    plt.subplot(3, 2, 5)
    plt.imshow(cv2.cvtColor(shape_img, cv2.COLOR_BGR2RGB))
    plt.title("5: Analyze object")

    plt.subplot(3, 2, 6)
    plt.imshow(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
    for el in top:
        for l in el:
            plt.scatter(el[0][0], el[0,1], s=20, c="blue")
    for el in bottom:
        for l in el:
            plt.scatter(el[0][0], el[0,1], s=20, c="magenta")
    for el in center_v:
        for l in el:
            plt.scatter(el[0][0], el[0,1], s=20, c="red")
    plt.title("6: Pseudolandmarks")

    # add overall title and adjust it so that it doesn't overlap with subplot titles
    plt.suptitle(image_path)
    # plt.subplots_adjust(top=0.85)

    plt.tight_layout()
    # display subplots
    plt.show()

    # if histogram:
    #     pass

def transform_directory(dir_src, dir_dst):
    pass


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        logger.error("Usage: python Transformation.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    if not os.path.isfile(image_path):
        logger.error(f"Image file not found: {image_path}")
        sys.exit(1)
    transform_image_1(image_path)
    transform_image(image_path, gaussian=True, mask=True, roi=True, analyze=True, pseudolandmarks=True, histogram=True)
    dir_src = "/Users/air/Documents/ecole/leaffliction/L__git/leaffliction_computer_vision/augmented_directory/Apple_Black_rot"
    dir_dst="/Users/air/Documents/ecole/leaffliction/L__git/leaffliction_computer_vision/transformed_dir"
    transform_directory(dir_src, dir_dst)

    # args = parser.parse_args()
    # print(args)
    # if (args.img_src):
    #     transform_image(args.img_src) \
    #         # if (args.mask):
    #     #     transform_image(args.img_src) # , args.dst)
    # if (args.src):
    #     print("no src")
    #     pass
    #     # ,args.gaussian,args.mask, args.roi, args.analyze, args.pseudolandmarks, args.histogram)  # , args.dst)

# Example usage
# display_image_transformations("images/Apple/apple_healthy/image (1).JPG")
# colors in lab !!!!
