#!/usr/bin/env python3

import os
import hashlib
from pathlib import Path
from Augmentation import augment_image
import logging
import shutil
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
    my_list = []
    for f in os.scandir(directory):
        # check if it is a jpg file
        if os.path.abspath(f).lower().endswith(".jpg"):
            hash_im = hashing_images(f)
            my_list.append((os.path.abspath(f), hash_im))
    # return pair(path_of_im, hash_of_im)
    return my_list


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
    unique_hash = []
    for elem in my_list:
        # print("elem", elem[1])
        if elem[1] in temp:
            # print(elem[1])
            logger.info(f"Removing image: {elem[0]}")
            os.remove(elem[0])
            # exit()
        else:
            temp.append(elem[1])
            unique_hash.append(elem)
    counts = count_jpg_files_pathlib(directory)
    print(f"  number of unique files: {counts}")
    return unique_hash


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
    print("________QUOT", quot, "REM", rem)
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
                print(f"quot {quot}")


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

    # No_of_files = len(os.listdir(directory))
    logger.info(f"Analysing images in our data folder: "
                f"{os.listdir(dest_dir)}")
    m = len(list(path.rglob("*")))
    logger.info(f"There are : {m} files and folders")

    subfolders = [f.path for f in os.scandir(dest_dir) if f.is_dir()]
    print("folders: ", len(subfolders))
    max_count = 0
    hash_all = []
    all_count = 0
    for mini_folder in subfolders:
        logger.info(f'Counting file in {mini_folder} directory')
        input("By continuing you agree to delete duplicate files"
              " if they exist. Enter anything.")
        hash_one_folder = removing_non_unique_elem(mini_folder)
        # print(hash_unique)
        current_len = count_jpg_files_pathlib(mini_folder)
        # print(hash_unique)
        l_hash = len(hash_one_folder)
        all_count += l_hash
        hash_all += hash_one_folder
        if current_len > max_count:
            max_count = current_len
    print(len(hash_all))
    print("Total for the entire dataset:")
    print(f"There're  {len(hash_all)} unique files.")
    print(f"maximum of files in folder: {max_count}")

    logger.info("Let's balance our data if necessary")
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
# проверка на похожие среди всего датасета
