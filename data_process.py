from enum import unique
import json
import os

"""
NOTES
1 --> There are 21 unique strokes (20 normal strokes + the empty stroke) - from running the scripts
2 --> There are 31 unique chars - from the paper
3 --> The data is NOT annotated on char level
"""


UNIQUE_STROKES_ORG = {'': 0, 'س': 0, 'ء': 0, 'ر': 0, 'ه': 0, 'م': 0, 'ع': 0, 'ٮ': 0, 'ں': 0, 'ص': 0, 'ٯ': 0, 'و': 0, '.': 0, 'ح': 0, 'ى': 0, 'د': 0, 'ﻛ': 0, 'ﺻ': 0, '\u0605': 0, 'ا': 0, 'ل': 0}


def create_labels(folder):
    path = 'dataset/dataset/'+folder+'/'
    files = os.listdir(path)

    for file in files:
        unique_strokes = UNIQUE_STROKES_ORG.copy()
        with open(path + file) as json_file:
            data = json.load(json_file)
            for str_dict in data:
                for key in str_dict.keys():
                    unique_strokes[key] = 1

            one_hot_vector = [1 if unique_strokes[char] else 0 for char in unique_strokes]

        label_file =  open('dataset/dataset/label/'+folder+'/' + file, 'w')
        label_file.write(str(one_hot_vector))
        label_file.close()


def main():
    create_labels(folder="test")
    create_labels(folder="train")
    create_labels(folder="valid")


if __name__ == "__main__":
    main()