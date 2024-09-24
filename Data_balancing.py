#!/usr/bin/env python3

import os
import hashlib
from pathlib import Path
from Augmentation import augment_image
import logging
import shutil
import numpy as np
import pandas as pd
# from PIL import Image


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# return hash of an image
def hashing_images(img_path):
    # This function will return the `md5` checksum for any input image.
    # rb -read binary
    with open(img_path, "rb") as f:
        img_hash = hashlib.md5()
        while chunk := f.read(8192):
            img_hash.update(chunk)
    return img_hash.hexdigest()


def hashing_directory(directory):
    path_and_hash = []
    for f in os.scandir(directory):
        # check if it is a jpg file
        if os.path.abspath(f).lower().endswith(".jpg"):
            hash_im = hashing_images(f)
            path_and_hash.append((os.path.abspath(f), hash_im))
    # return pair(path_of_im, hash_of_im)
    return path_and_hash


def count_jpg_files_pathlib(directory):
    path = Path(directory)
    i = 0
    for file in path.iterdir():
        if os.path.abspath(file).lower().endswith(".jpg"):
            i += 1
    return i


def removing_non_unique_elem(directory):
    counts = count_jpg_files_pathlib(directory)
    my_list = hashing_directory(directory)

    temp = []
    path_hash = []
    for elem in my_list:
        if elem[1] in temp:
            logger.info(f"Removing image: {elem[0]}")
            os.remove(elem[0])
        else:
            temp.append(elem[1])
            path_hash.append(elem)

    counts = count_jpg_files_pathlib(directory)
    print(f"  number of unique files: {counts}")
    return path_hash


# balance your data set
def adding_new_file(how_many_add, directory):
    logger.info(f"We need to add : {how_many_add} "
                f"new files in this '{directory}' directory ")
    images = sorted(os.listdir(directory))
    # subfolders = [f.path for f in os.scandir(directory) if f.is_dir()]
    if how_many_add >= 6:
        quot, rem = divmod(how_many_add, 6)
    if how_many_add < 6:
        quot = 0
        rem = how_many_add
    for img in images:
        if img.lower().endswith(".jpg"):
            img_path = directory + '/' + img
            if quot == 0 and rem > 0:
                augment_image(os.path.abspath(img_path), rem)
                rem = 0
                break
            if quot:
                augment_image(os.path.abspath(img_path), 6)
                quot -= 1


def data_balancing(directory):
    logger.info("Creating an augmented_directory folder")
    # defining source and destination
    # paths
    src_dir = directory
    # os.mkdir("./augmented_directory")
    dest_dir = "augmented_directory"
    # getting all the files in the source directory
    # files = os.listdir(src_dir)
    shutil.copytree(src_dir, dest_dir)

    path = Path(dest_dir)
    logger.info(f"Analysing images in our data folder: "
                f"{os.listdir(dest_dir)}")
    m = len(list(path.rglob("*")))
    logger.info(f"There are : {m} files and folders")

    subfolders = [f.path for f in os.scandir(dest_dir) if f.is_dir()]
    print("folders: ", len(subfolders), '\n')

    path_hash_all = []
    for mini_folder in subfolders:
        logger.info(f'Check if ther are any duplicated images '
                    f'in {mini_folder} directory')
        input("By continuing you agree to delete duplicate files"
              " in this directory if they exist. Enter anything.")
        path_hash_folder = removing_non_unique_elem(mini_folder)
        path_hash_all += path_hash_folder
        print('\n')

    logger.info("let's see if there are any similar pictures between classes")
    only_hash = []
    input("By continuing you agree to delete all similar images "
          "between classesif they exist. Enter anything.")
    for p in path_hash_all:
        only_hash.append(p[1])
    indices = np.where(pd.Series(only_hash).duplicated(keep=False))[0]

    for elem in indices:
        logger.info(f"Removing image: {path_hash_all[elem][0]}")
        os.remove(path_hash_all[elem][0])

    max_count = 0
    all_img = 0
    for mini_folder in subfolders:
        current_len = count_jpg_files_pathlib(mini_folder)
        all_img += current_len
        if current_len > max_count:
            max_count = current_len
        logger.info(f'Counting file in {mini_folder} '
                    f'directory: {current_len} images')
    print('\n')
    print("Total for the entire dataset:")
    print(f"There're  {all_img} unique files.")
    print(f"maximum of files in folder: {max_count}")
    print('\n')
    logger.info("Let's balance our data if necessary")
    input("Please enter anything to continue.")
    for mini_folder in subfolders:
        im_count = count_jpg_files_pathlib(mini_folder)
        logger.info(f'Files in {mini_folder} directory: {im_count}')
        how_many_add = max_count - im_count
        if how_many_add > 0:
            print(f"You need to add: {how_many_add} files in this directory")
            # print(f"Need to add {how_many_add} in folder{mini_folder}")
            adding_new_file(how_many_add, mini_folder)
        else:
            pass
    logger.info("Data balancing completed.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        logger.error("Usage: python Data_balancing.py <data_path>")
        sys.exit(1)
    data_path = sys.argv[1]
    if not os.path.isdir(data_path):
        logger.error(f"Directory with data was not found: {data_path}")
        sys.exit(1)
    data_balancing(data_path)

# исправить рандомность.
# если больше чем н на 6 умноженное
