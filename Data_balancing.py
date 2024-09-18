import os
import hashlib
from pathlib import Path
from Augmentation import augment_image
import logging
# from PIL import Image


# class ColoredFormatter(logging.Formatter):
#     COLORS = {'DEBUG': '\033[94m', 'INFO': '\033[92m', 'WARNING': '\033[93m',
#               'ERROR': '\033[91m', 'CRITICAL': '\033[95m'}
#
#     def format(self, record):
#         log_fmt = f"{self.COLORS.get(record.levelname, '')}" \
#                   f"%(asctime)s - %(levelname)s - %(message)s\033[0m"
#         formatter = logging.Formatter(log_fmt)
#         return formatter.format(record)
#
#
# # logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger()
#
#
# # logger.setLevel(logging.DEBUG)
# # create console handler with a higher log level
# ch = logging.StreamHandler()
#
# # ch.setLevel(logging.DEBUG)
# ch.setFormatter(ColoredFormatter())
# logger.addHandler(ch)
#
# logger.debug("debug message")
# logger.info("info message")
# logger.warning("warning message")
# logger.error("error message")
# logger.critical("critical message")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#
# def count_files(directory):
#     a = len([name for name in os.listdir(directory) if
#     os.path.isfile(os.path.join(directory, name))])
#     return len(list(folder.rglob("*")))


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
    list_hash = []
    list_path = []
    my_list = []
    for f in os.scandir(directory):
        # check if it is a jpg file
        if os.path.abspath(f).lower().endswith(".jpg"):
            hash_im = hashing_images(f)
            list_hash.append(hash_im)
            list_path.append(os.path.abspath(f))
            my_list.append((os.path.abspath(f), hash_im))
    return list_hash, list_path, my_list


def count_jpg_files_pathlib(directory):
    path = Path(directory)
    # print(f"path {directory}:")
    i = 0
    for file in path.iterdir():
        # print(type(os.path.abspath(file)))
        if os.path.abspath(file).lower().endswith(".jpg"):
            i += 1
    return i


def removing_non_unique_elem(directory):
    my_folder = Path(directory)
    counts = count_jpg_files_pathlib(directory)
    list_hash, list_path, my_list = hashing_directory(directory)
    len_hash = len(list_hash)
    hash_unique = list(dict.fromkeys(list_hash))
    itog = []
    for elem in my_list:
        # print("elem", elem[1])
        if elem[1] in itog:
            # print(elem[1])
            logger.info(f"Removing image: {elem[0]}")
            os.remove(elem[0])
            # exit()
        else:
            itog.append(elem[1])
    # print(f"  number of files: {len_hash}")
    print(f"  number of unique files: {len(hash_unique)}")
    # print(f"  number of total files: {len(itog)}")
    return counts, hash_unique


# balance your data set
def adding_new_file(how_many_add, directory):
    logger.info(f"We need to add : {how_many_add} "
                f"new files in this '{directory}' directory ")
    images = sorted(os.listdir(directory))
    # subfolders = [f.path for f in os.scandir(directory) if f.is_dir()]
    if how_many_add >= 10:
        quot, rem = divmod(how_many_add, 10)
    if how_many_add < 10:
        quot = 0
        rem = how_many_add
    # print("________QUOT", quot, "REM", rem)
    for img in images:
        if img.lower().endswith(".jpg"):
            img_path = directory + '/' + img
            if quot == 0 and rem > 0:
                augment_image(os.path.abspath(img_path), rem)
                rem = 0
                break
            if quot:
                augment_image(os.path.abspath(img_path), 10)
                quot -= 1
                print(f"quot {quot}")


def data_balancing(directory):
    path = Path(directory)
    # No_of_files = len(os.listdir(directory))
    logger.info(f"Analysing images in our data folder: "
                f"{os.listdir(directory)}")
    m = len(list(path.rglob("*")))
    logger.info(f"There are : {m} files and folders")

    subfolders = [f.path for f in os.scandir(directory) if f.is_dir()]
    print("folders: ", len(subfolders))
    max_count = 0
    hash_all = []
    all_count = 0
    for mini_folder in subfolders:
        logger.info(f'Counting file in {mini_folder} directory')
        current_len = count_jpg_files_pathlib(mini_folder)
        len_files, hash_unique = removing_non_unique_elem(mini_folder)
        all_count += len_files
        hash_all += hash_unique
        l_hash = len(hash_unique)
        # print("L:", l_hash)
        # print("MAX:", max_count)
        if current_len > max_count:
            max_count = current_len
        # print(f"MAX: {max_count}")
    hash_all_unique = list(dict.fromkeys(hash_all))
    print("Total for the entire dataset:")
    print(f"There're  {len(hash_all_unique)} unique files.")
    print(f"non unique all: {len(hash_all)}")
    # print("non unique at all: ", len(hash_all))
    print(f"maximum of files in folder: {max_count}")
    print(f"all files: {all_count}")

    # for mini_folder in subfolders:
    #     print(f'Counting file in {mini_folder} directory')
    #     print(count_jpg_files_pathlib(mini_folder))
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
    # data_path = "/Users/air/Documents/ecole/leaffliction/images/"
    data_balancing(data_path)

# if __name__ == "__main__":
#     directory =
#     "/Users/air/Documents/ecole/leaffliction/images/Apple_Black_rot"
#     removing_non_unique_elem(directory)
#     adding_new_file(10, directory)

# исправить рандомность.
# проверка на похожие среди всего датасета
