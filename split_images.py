import os
from turtle import width
from PIL import Image

from data_label import main


#---------Global variable-----------#
# PATH = 'dataset/images/train/'
# PATH = 'dataset/images/test/'
PATH = 'dataset/images/val/'
FOLDERS = ["Kufi", "Farisi", "Thuluth", "Diwani"]

def split_img():

    for folder in FOLDERS:

        file_path = PATH + folder + "/"
        files = os.listdir(file_path)
        files = [file for file in files if os.path.isfile(file_path + file)] # ignore the folders
        
        for file in files:

            if file == '.DS_Store': # if the file is meta file, ignore it
                continue
            img = Image.open(file_path + file)
            width, height = img.size

            (left_img1, upper_img1) = (0, 0)
            (right_img1, lower_img1) = (width // 2, height)
            img1 = img.crop((left_img1, upper_img1, right_img1, lower_img1))
            img1.save(file_path + "Hand_" + file)

            (left_img2, upper_img2) = (width // 2, 0)
            (right_img2, lower_img2) = (width, height)
            img2 = img.crop((left_img2, upper_img2, right_img2, lower_img2))
            img2.save(file_path + "Digital_" + file)

            os.remove(file_path + file)

def main():
    split_img()


if __name__ == "__main__":
    main()

        