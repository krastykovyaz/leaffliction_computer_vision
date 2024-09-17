import os
import glob
import hashlib
from pathlib import Path
from Augmentation import augment_image
from PIL import Image
#
# def count_files(directory):
#     a = len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])
#     return len(list(folder.rglob("*")))

#return hash of an image
def hashing_images(img_path):
    # This function will return the `md5` checksum for any input image.
    #rb -read binary
    with open(img_path, "rb") as f:
        img_hash = hashlib.md5()
        while chunk := f.read(8192):
           img_hash.update(chunk)
    return img_hash.hexdigest()

def hashing_folder(directory):
    list_hash = []
    list_path = []
    my_list = []
    for f in os.scandir(directory):
        #check if it is a jpg file
        if os.path.abspath(f).lower().endswith(".jpg"):
            hash_im =  hashing_images(f)
            list_hash.append(hash_im)
            list_path.append(os.path.abspath(f))
            my_list.append((os.path.abspath(f), hash_im))
    return list_hash, list_path, my_list

def count_jpg_files_pathlib(directory):
    path = Path(directory)
    print(f"path {directory}:")
    i = 0
    for file in path.iterdir():
        # print(type(os.path.abspath(file)))
        if os.path.abspath(file).lower().endswith(".jpg"):
            i+=1
    return i

def removing_non_unique_elem(directory):
    my_folder = Path(directory)
    counts = count_jpg_files_pathlib(directory)
    list_hash, list_path, my_list = hashing_folder(directory)
    len_hash = len(list_hash)
    hash_unique = list(dict.fromkeys(list_hash))
    itog = []
    for elem in my_list:
        # print("elem", elem[1])
        if elem[1] in itog:
            # print(elem[1])
            print("----revoming ", elem[0])
            os.remove(elem[0])
            # exit()
        else:
            itog.append(elem[1])
    print(f"  number of files: {len_hash}")
    print(f"  number of unique files: {len(hash_unique)}")
    print(f"  number of total files: {len(itog)}")
    return counts, hash_unique

# balance your data set
def adding_new_file(how_many_add, directory):
    print(f"_We need to add : {how_many_add} new files in directory {directory}")
    images = sorted(os.listdir(directory))
    # subfolders = [f.path for f in os.scandir(directory) if f.is_dir()]
    print(images)
    if how_many_add >= 10 :
        quot, rem = divmod(how_many_add, 10)
    if how_many_add < 10:
        quot = 0
        rem = how_many_add
    # print("________QUOT", quot, "REM", rem)
    for img in images:
        if img.lower().endswith(".jpg"):
            img_path = directory +'/'+ img
            print("++++++++++++++++",type(img_path))
            if quot == 0 and rem > 0:
                print(f"________Augmentation {img}")
                augment_image(os.path.abspath(img_path), rem)
                rem = 0
                break
            if quot:
                print(f"________Augmentation of image 10{img}")
                augment_image(os.path.abspath(img_path), 10)
                quot-=1
                print(f"quot {quot}")

# def count_umber_files():


if __name__ == "__main__":
    directory = "/Users/air/Documents/ecole/leaffliction/images/"
    path = Path(directory)
    counts = count_jpg_files_pathlib(path)
    # print(f"Количество файлов всех файлов в папке: {counts}")
    # my_path, dirs, files = next(os.walk(directory))
    # print(f"Количество файлов всех файлов в папке: {my_path}")
    No_of_files = len(os.listdir(directory))

    print(f"Количество файлов всех файлов в папке1: {No_of_files}")
    pngCounter = len(glob.glob1(directory, "*.png"))
    print(f"Количество файлов всех файлов в папке png: {pngCounter}")
    print(f"Количество файлов всех файлов в папке  3333: {os.listdir(directory)}")
    m = len(list(path.rglob("*")))
    print(f"Количество файлов всех файлов в папке  3333: {m}")


    subfolders = [f.path for f in os.scandir(directory) if f.is_dir()]
    print(subfolders)
    print("folders: ", len(subfolders))
    max_count = 0
    hash_all = []
    all_count = 0
    m = 1
    for mini_folder in subfolders:
        print(f'Counting file in {mini_folder} directory')
        # max_count_old = max_count_new
        current_len = count_jpg_files_pathlib(mini_folder)
        # print(f"max {max_count_new}")
        len_files, hash_unique = removing_non_unique_elem(mini_folder)
        all_count += len_files
        hash_all +=hash_unique
        l = len(hash_unique)
        print("L:", l)
        print("MAX:", max_count)
        # print(l)
        print(current_len)
        if current_len > max_count:
            max_count = current_len
        print("MAX:", max_count)
        print("AAAAAAAAA:", m)
        m+=1
    hash_all_unique = list(dict.fromkeys(hash_all))
    print("Total for the entire dataset:")
    print(f"There're  {len(hash_all_unique)} unique files.")
    print("non unique all: ", len(hash_all))
    # print("non unique at all: ", len(hash_all))
    print("maximum of files in folder: " , max_count)
    print("all files: ", all_count)

    # for mini_folder in subfolders:
    #     print(f'Counting file in {mini_folder} directory')
    #     print(count_jpg_files_pathlib(mini_folder))
    print("ADDING___________________")
    for mini_folder in subfolders:
        im_count= count_jpg_files_pathlib(mini_folder)
        print(f'Files in {mini_folder} directory: {im_count}')
        how_many_add = max_count - im_count
        print(f"you need to add: {how_many_add}")
        if how_many_add > 0:
            print(f"HOWWW many add {how_many_add} in folder{mini_folder}")
            # adding_new_file(how_many_add, mini_folder)
        else:
            pass

#
# if __name__ == "__main__":
#     directory = "/Users/air/Documents/ecole/leaffliction/images/Apple_Black_rot"
#     removing_non_unique_elem(directory)
#     adding_new_file(10, directory)


#удалить все файлы не пнг
#если будет очень маленький набор файлов. То наша аугментация не поможет.
#проверка на похожие всего датасета


