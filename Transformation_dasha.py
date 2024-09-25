#!/usr/bin/env python3

from plantcv import plantcv as pcv
from pilkit.lib import Image
import logging
from argparse import ArgumentParser
from argparse import ArgumentTypeError
from matplotlib import pyplot as plt
import matplotlib
import cv2
import os
from pathlib import Path
# import pandas as pd
# import numpy as np
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


# Example usage
# plot_color_histogram("images/Apple/apple_healthy/image (1).JPG")


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def dir_src_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise ArgumentTypeError(f"readable_dir:{path} is not a valid path")


def dir_dst_path(path):
    if os.path.isdir(path):
        return path
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception as e:
            logger.error(f"Error making directory {path}: {e}")


def file_img(img):
    # hsv_img = matplotlib.colors.rgb_to_hsv(np_img)
    # check if it is a jpg file ???????????????????????
    if os.path.abspath(img).lower().endswith(".jpg"):
        try:
            Image.open(img)
        except IOError:
            raise ArgumentTypeError(f"{img} is not a valid image")
        # filename not an image file
        return img
    else:
        raise ArgumentTypeError(f"{img} is not a valid image")


parser = ArgumentParser(description=__doc__)
#

parser.add_argument("img_src", help="Path to the input image",
                    type=file_img, nargs='?')

parser.add_argument(
    "-src",
    # dest="first_string",
    help="the source directory of images",
    type=dir_src_path
)

parser.add_argument(
    "-dst",
    # dest="second_string",
    help="destinagtion directory for transformed images",
    type=dir_dst_path
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


class Transform_image:
    def __init__(self, image_path, display=True, dir_dst=None,
                 gaussian=False, mask=False, roi=False,
                 analyze=False, pseudolandmarks=False, histogram=False):
        # # I can read image also like this:
        # self.path = image_path
        self.img = Image.open(image_path)
        # # Transform to numpy array
        # self.np_img = np.asarray(self.img)

        # But this is less code of reading image
        self.np_img, self.path, path_name = pcv.readimage(filename=image_path,
                                                          mode="native")
        print("Path", self.path)
        self.name = Path(path_name).stem
        self.tr_mask = None

        self.dir_dst = dir_dst
        self.display = display

        self.gaussian_f = gaussian
        self.mask_f = mask
        self.roi_f = roi
        self.analyze_f = analyze
        self.pseudolandmarks_f = pseudolandmarks
        self.histogram_f = histogram

        logger.info(f"Let's transform image: {self.path}")
        self.transformation()
        print('\n')

    # Histogram Analyze Color using plantcv
    def one_channel(self, chans, colors, labels):
        for (chan, color) in zip(chans, colors):
            hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
            plt.plot(hist, color=color, label=labels)
            plt.xlim([0, 256])

    def plot_color_histogram(self):
        image = cv2.imread(self.path)

        # Transform to numpy array
        hsv_img = cv2.cvtColor(self.np_img, cv2.COLOR_BGR2HSV)
        lab_img = cv2.cvtColor(self.np_img, cv2.COLOR_BGR2LAB)
        chans = cv2.split(self.np_img)
        chans_hsv = cv2.split(hsv_img)
        chans_lab = cv2.split(lab_img)
        plt.figure()
        plt.title("Color Histogram")
        plt.xlabel("Bins")
        plt.ylabel("# of Pixels")
        self.one_channel(chans[0], "b", "blue")
        self.one_channel(chans[1], "g", "green")
        self.one_channel(chans[2], "r", "red")
        self.one_channel(chans_hsv[0], "m", "hue")
        self.one_channel(chans_hsv[1], "c", "saturation")
        self.one_channel(chans_hsv[2], "y", "value")
        self.one_channel(chans_lab[0], "b", "lightness")
        self.one_channel(chans_lab[1], "c", "green_red")
        self.one_channel(chans_lab[2], "y", "blue_yellow")
        plt.legend()
        if self.display:
            plt.show()
        else:
            histogram_name_1 = self.dir_dst + \
                               '/' + self.name + "_7_histogram.png"
            plt.savefig(histogram_name_1)
            logger.info(f"saving image {histogram_name_1}")

    # Examine signal distribution within an image
    def pcv_histogram(self):
        # prints out an image histogram of signal within image
        hist_figure, hist_data = pcv.visualize.histogram(img=self.np_img,
                                                         title="histogram",
                                                         mask=self.tr_mask,
                                                         hist_data=True)
        # image = cv2.imread(self.path)
        hsv_img = matplotlib.colors.rgb_to_hsv(self.np_img)
        # hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist_figure_hsv, hist_data_hsv = \
            pcv.visualize.histogram(img=hsv_img, mask=self.tr_mask,
                                    hist_data=True)
        if self.histogram_f and self.dir_dst:
            histogram_name = self.dir_dst + \
                             '/' + self.name + "_6_histogram_rgb.png"
            pcv.print_image(hist_figure, filename=histogram_name)
            logger.info(f"saving image {histogram_name}")

    # Roi objects (Region of interest to mask)
    def pcv_roi(self):
        self.center_x = int(self.np_img.shape[0] / 2)
        self.center_y = int(self.np_img.shape[1] / 2)
        self.radius = int((self.center_x + self.center_y) / 2)
        roi = pcv.roi.circle(img=self.np_img, x=self.center_x,
                             y=self.center_y, r=75)
        # Make a new filtered mask that only keeps the plant in your ROI
        # and not objects outside of the ROI
        # We have set to partial here so that if a leaf extends outside of
        # your ROI it will still be selected.
        filtered_mask = pcv.roi.filter(mask=self.tr_mask,
                                       roi=roi, roi_type='partial')
        if self.roi_f and self.dir_dst:
            roi_name = self.dir_dst + '/' + self.name + "_3_roi.png"
            pcv.print_image(filtered_mask, roi_name)
            logger.info(f"saving image {roi_name}")

        self.tr_mask = filtered_mask

    # Characterize object shapes
    def pcv_analyze_object(self):
        shape_img = pcv.analyze.size(img=self.np_img,
                                     labeled_mask=self.tr_mask,
                                     n_labels=1)
        if self.analyze_f and self.dir_dst:
            shape_name = self.dir_dst + \
                         '/' + self.name + "_4_analyze_object.png"
            pcv.print_image(shape_img, shape_name)
            logger.info(f"saving image {shape_name}")
        return shape_img

    # 6 Identify a set of land mark points
    def pcv_pseudolanmarks(self):
        # device = 1
        # Results in set of point values that may indicate tip points
        top, bottom, center_v = \
            pcv.homology.x_axis_pseudolandmarks(img=self.np_img,
                                                mask=self.tr_mask)
        pseudo_img = pcv.apply_mask(img=self.np_img,
                                    mask=top,
                                    mask_color='black')
        return top, bottom, center_v

    def color_space_image(self):
        # Update params related to plotting so we can see better
        pcv.params.text_size = 5
        pcv.params.text_thickness = 5
        # Look at the colorspace - which of these looks the best for masking?
        # Which channel makes the plant look most distinct from the background?
        colorspace_img = pcv.visualize.colorspaces(rgb_img=self.np_img,
                                                   original_img=False)
        if self.dir_dst:
            color_sp_name = \
                self.dir_dst + '/' + self.name + "_0_colorspace.png"
            pcv.print_image(colorspace_img, color_sp_name)
            logger.info(f"saving image {color_sp_name}")

    def transformation(self):
        self.color_space_image()
        # # Optionally, set a sample label name
        # pcv.params.sample_label = "plant"
        if self.dir_dst:
            orig_name = self.dir_dst + '/' + self.name + "_0_original.png"
            pcv.print_image(self.np_img, orig_name)
            logger.info(f"saving image {orig_name}")

        # Thresholding method - image converted from RGB to gray.
        gray_img = pcv.rgb2gray(rgb_img=self.np_img)
        # Create binary image from a gray image based on threshold values,
        threshold_light = pcv.threshold.binary(gray_img=gray_img,
                                               threshold=115,
                                               object_type='dark')
        # Noise Reduction
        # delete small parts of pixels
        bsa_fill1 = pcv.fill(bin_img=threshold_light,
                             size=25)
        self.tr_mask = bsa_fill1
        # bsa_fill1 = pcv.dilate(gray_img=bsa_fill1, ksize=1.5, i = 2)
        # closing small pixels
        filled_mask1 = pcv.closing(gray_img=self.tr_mask)
        # pcv.print_image(filled_mask1, "1_binary.png")

        # 2 Gaussian blur
        # Apply gaussian blur to a binary image
        gaussian_img = pcv.gaussian_blur(img=self.tr_mask,
                                         ksize=(3, 3),
                                         sigma_x=0,
                                         sigma_y=None)
        # img_flip.save(f"{img_name}_Flip{img_extension}")
        if self.gaussian_f and self.dir_dst:
            gauss_name = self.dir_dst + '/' + self.name + "_1_gaussian.png"
            pcv.print_image(gaussian_img, gauss_name)
            logger.info(f"saving image {gauss_name}")

        # #3 Mask - Apply binary 'white' mask over an image.
        masked_img = pcv.apply_mask(img=self.np_img,
                                    mask=self.tr_mask,
                                    mask_color='white')
        if self.mask_f and self.dir_dst:
            masked_name = self.dir_dst + '/' + self.name + "_2_masked.png"
            pcv.print_image(masked_img, masked_name)
            logger.info(f"saving image {masked_name}")

        self.pcv_roi()

        if self.analyze_f and self.dir_dst:
            shape_img = self.pcv_analyze_object()

        if self.pseudolandmarks_f or self.display:
            top, bottom, center_v = self.pcv_pseudolanmarks()

        if self.histogram_f and self.dir_dst:
            self.pcv_histogram()
            self.plot_color_histogram()

        if self.display:
            # Display results
            plt.figure(figsize=(10, 8))
            # plt.title(figure_title, y=1.08)

            plt.subplot(3, 2, 1)
            plt.imshow(cv2.cvtColor(self.np_img, cv2.COLOR_BGR2RGB))
            plt.title("1: Original")

            plt.subplot(3, 2, 2)
            plt.imshow(cv2.cvtColor(gaussian_img, cv2.COLOR_BGR2RGB))
            plt.title("2: Gaussian Blur")

            plt.subplot(3, 2, 3)
            plt.imshow(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
            plt.title("3: Analyzed Object")

            fig = plt.subplot(3, 2, 4)
            plt.imshow(cv2.cvtColor(self.np_img, cv2.COLOR_BGR2RGB))
            circle = plt.Circle((self.center_x, self.center_y),
                                self.radius*0.8, alpha=1,
                                fill=None, linewidth=5,
                                facecolor=None, linestyle='-',
                                edgecolor="blue")
            plt.scatter(self.center_x, self.center_y, 10, facecolors='none',
                        edgecolors='blue')
            fig.add_patch(circle)
            plt.title("4: ROI objects")

            plt.subplot(3, 2, 5)
            plt.imshow(cv2.cvtColor(shape_img, cv2.COLOR_BGR2RGB))
            plt.title("5: Analyze object")

            plt.subplot(3, 2, 6)
            plt.imshow(cv2.cvtColor(self.np_img, cv2.COLOR_BGR2RGB))
            for t in top:
                plt.scatter(t[0][0], t[0, 1], s=20, c="blue")
            for bt in bottom:
                plt.scatter(bt[0][0], bt[0, 1], s=20, c="magenta")
            for cn in center_v:
                plt.scatter(cn[0][0], cn[0, 1], s=20, c="red")
            plt.title("6: Pseudolandmarks")

            # add overall title and adjust it
            # so that it doesn't overlap with subplot titles
            plt.suptitle(self.path)
            # plt.subplots_adjust(top=0.85)

            plt.tight_layout()
            # display subplots
            plt.show()

            # if histogram:
            #     pass


def transform_directory(dir_src, dir_dst, gaussian=False,
                        mask=False, roi=False, analyze=False,
                        pseudolandmarks=False, histogram=False):
    for f in os.scandir(dir_src):
        # check if it is a jpg file
        if os.path.abspath(f).lower().endswith(".jpg"):
            try:
                print("image open")
                Image.open(f)
            except:
                pass
            Transform_image(f, display=False, dir_dst=dir_dst,
                            gaussian=gaussian,
                            mask=mask, roi=roi,
                            analyze=analyze,
                            pseudolandmarks=pseudolandmarks,
                            histogram=histogram)


if __name__ == "__main__":
    args = parser.parse_args()
    print("args: ", args)
    print()
    if (args.img_src):
        tr_im = Transform_image(args.img_src)
    elif (args.src and args.dst):
        transform_directory(args.src, args.dst, args.gaussian,
                            args.mask, args.roi, args.analyze,
                            args.pseudolandmarks, args.histogram)

    else:
        exit("Welcome to image trandformation!"
             "Enter as argument '-h' to know how to use it")
        # args = parser.parse_args("-help")
# Example usage
# display_image_transformations("images/Apple/apple_healthy/image (1).JPG")
# colors in lab !!!!
# only jpg files?
