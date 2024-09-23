#!/usr/bin/env python3

from plantcv import plantcv as pcv
from pilkit.lib import Image
import logging
from argparse import ArgumentParser
import cv2
import numpy as np
from matplotlib import pyplot as plt
# from PIL import Image, ImageOps, ImageEnhance, ImageFilter
# from pilkit.processors import TrimBorderColor
# import os


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


# Example usage
# display_image_transformations("images/Apple/apple_healthy/image (1).JPG")
def plot_color_histogram(image_path):
    image = cv2.imread(image_path)
    chans = cv2.split(image)
    colors = ("b", "g", "r")
    plt.figure()
    plt.title("Color Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")

    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])

    plt.show()

# Example usage
# plot_color_histogram("images/Apple/apple_healthy/image (1).JPG")


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


parser = ArgumentParser(description=__doc__)
#

parser.add_argument("img_src", help="transform one image")

parser.add_argument(
    "-src",
    # dest="first_string",
    help="the source directory of images",
    # required=True,
)

parser.add_argument(
    "-dst",
    # dest="second_string",
    help="destinagtion directory for transformed images",
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
# pcv.params.debug = "print"


# def read_image(image_path):
#     #Read image
#     img = Image.open(image_path)
#     #Transform to numpy array
#     np_img = np.asarray(img)
#     # img_name, img_extension = os.path.splitext(image_path)


def transform_image(image_path, gaussian, mask,
                    roi, analyze, pseudolandmarks, histogram):
    print(image_path)
    # I can read image also like this:
    # Read image
    img = Image.open(image_path)

    # Transform to numpy array
    np_img = np.asarray(img)

    # #But this is less code
    # # read in image
    # np_img, path, img_filename = pcv.readimage(filename=image_path,
    #                                            mode="native")
    # print(path)
    # print(img_filename)

    # Optionally, set a sample label name
    pcv.params.sample_label = "plant"

    # 1 Original
    # pcv.print_image(np_img, "0_original.png")

    # ###1. Methods of Isolating Target Objects
    # ####Object Segmentation Approaches
    # ####Thresholding method
    # 2_1 Thresholded image from gray
    # image converted from RGB to gray.
    gray_img = pcv.rgb2gray(rgb_img=np_img)
    # Create binary image from a gray image based on threshold values,
    # targeting light objects in the image.
    # Perform thresholding to generate a binary image
    threshold_light = pcv.threshold.binary(gray_img=gray_img,
                                           threshold=120,
                                           object_type='dark')
    # delete small parts of pixels
    bsa_fill1 = pcv.fill(bin_img=threshold_light,
                         size=15)

    # bsa_fill1 = pcv.dilate(gray_img=bsa_fill1, ksize=1.5, i = 2)

    # closing small pixels
    filled_mask1 = pcv.closing(gray_img=bsa_fill1)
    # pcv.print_image(filled_mask1, "1_binary.png")

    # ####Noise Reduction
    # 2 Gaussian blur DONE
    # Apply gaussian blur to a binary image
    # that has been previously thresholded.
    gaussian_img = pcv.gaussian_blur(img=filled_mask1,
                                     ksize=(3, 3),
                                     sigma_x=0,
                                     sigma_y=None)
    # img_flip.save(f"{img_name}_Flip{img_extension}")
    if gaussian:
        pcv.print_image(gaussian_img, "2_gaussian_image.png")

    print("mask", type(filled_mask1))
    # #3 Mask
    # # Apply binary 'white' mask over an image.
    masked_img = pcv.apply_mask(img=np_img,
                                  mask=gaussian_img,
                                  mask_color='white')
    if mask:
        pcv.print_image(masked_img, "3_mask.png")

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
    if roi:
        pass
    # filtered_mask = pcv.roi.filter(mask=bsa_fill1,
    #                                roi=roi, roi_type='partial')
    # pcv.print_image(filtered_mask, "5_roi.png")

    # ###2. Object Analysis in PlantCV
    # #5 Analyze object - Analyze plant shape
    # Characterize object shapes
    shape_img = pcv.analyze.size(img=np_img,
                                   labeled_mask=bsa_fill1,
                                   n_labels=1)
    if analyze:
        logger.info("saving analyze picture")
        pcv.print_image(shape_img, "6.png")

    # 6 Pseudolandmarks
    device = 1
    # Identify a set of land mark points
    # Results in set of point values that may indicate tip points
    top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(img=np_img, mask=bsa_fill1)

    # pseudo_img1 = cv2.circle(np_img, top, radius=0, color=(0, 0, 255), thickness=-1)

    print("np", type(np_img))
    print("top", type(top))
    print("bottom", type(bottom))
    print("center", type(center_v))
    pseudo_img = pcv.apply_mask(img=np_img,
                                    mask=top,
                                    mask_color='black')
    if pseudolandmarks:
        pass
    if histogram:
        # 7 Histogram Analyze Color DONE
        # Examine signal distribution within an image
        # prints out an image histogram of signal within image
        logger.info("saving histogram")
        pcv.params.debug = "print"
        hist_figure, hist_data = pcv.visualize.histogram(img=np_img,
                                                         title="histogram",
                                                         mask=bsa_fill1,
                                                         hist_data=True)
        # hist_data.savefig("normalhistogram.png")
        pcv.params.debug = "None"
        # hist_figure.plot_hist('histogram')

    # # Green-Magenta ('a') channel is output
    # a_channel = pcv.rgb2gray_lab(rgb_img=np_img, channel='a')
    # hist_figure_a, hist_data_a = pcv.visualize.histogram(
    #     img=a_channel, mask=bsa_fill1, hist_data=True)

    # analysis_images = pcv.analyze.color(np_img, bsa_fill1)
    # pcv.print_image(analysis_images, "8_analysis_images.png")
    # pcv.outputs.save_results(filename="results")

    # Display results
    plt.figure(figsize=(10, 8))
    # plt.title(figure_title, y=1.08)

    plt.subplot(3, 2, 1)
    plt.imshow(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
    plt.title("1: Original")
    # plt.text(0.5, 1.08, "1: Original",
    #          horizontalalignment='center',
    #          fontsize=20)
    #
    print(type(np_img))
    plt.subplot(3, 2, 2)
    plt.imshow(cv2.cvtColor(gaussian_img, cv2.COLOR_BGR2RGB))
    plt.title("2: Gaussian Blur")

    plt.subplot(3, 2, 3)
    # plt.imshow(bsa_fill1)
    plt.imshow(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
    # # plt.imshow(cv2.cvtColor(analyze, cv2.COLOR_BGR2RGB))
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
    # plt.scatter(x, y, s=3, c="red", vmin=0, vmax=100)
    # plt.imshow(top)
    # print(top.shape)
    # print(top)
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
    # plt.imshow(pseudo_img, cmap='gray')
    plt.title("6: Pseudolandmarks")
    # add overall title and adjust it so that it doesn't overlap with subplot titles
    plt.suptitle(image_path)
    # plt.subplots_adjust(top=0.85)

    plt.tight_layout()
    # display subplots
    plt.show()

if __name__ == "__main__":
    # import sys
    # if len(sys.argv) != 2:
    #     logger.error("Usage: python Transformation.py <image_path>")
    #     sys.exit(1)
    # image_path = sys.argv[1]
    # if not os.path.isfile(image_path):
    #     logger.error(f"Image file not found: {image_path}")
    #     sys.exit(1)
    args = parser.parse_args()
    # transform_image(image_path)
    # display_image_transformations(args.img_src)  # , args.dst)

    print(args)
    if (args.img_src):
        transform_image(args.img_src, args.gaussian,
                        args.mask, args.roi, args.analyze,
                        args.pseudolandmarks, args.histogram)  # , args.dst)
        # if (args.mask):
        #     transform_image(args.img_src) # , args.dst)
    if (args.src):
        print("no src")
        pass
