import os
from tkinter.ttk import Style
import PIL.Image, PIL.ImageTk
from tkinter import *

PATH = 'dataset/pix2pix/test/'
# PATH = 'dataset/pix2pix/train/'
# PATH = 'dataset/pix2pix/val/'

def get_calligraphy_style(file):

    master = Tk()
    master.geometry("1000x1000")
    
    # Tkinter string variable with default value of 1
    v = StringVar(master, "1")
    
    # Dictionary to create multiple buttons
    values = {"Kufi" : "1",
            "Farisi" : "2",
            "Thuluth" : "3",
            "Diwani" : "4",
            "Other" : "5"}
    
    # for loop to create all the Radiobuttons
    for (text, value) in values.items():
        Radiobutton(master, text = text, variable = v,
                    value = value, indicator = 0,
                    selectcolor="red").pack(fill = X, ipady = 5)

    # a canvas to displayy the image
    canvas = Canvas(master, width=700, height=350)
    canvas.pack()
    img = PIL.ImageTk.PhotoImage(file = file)
    canvas.create_image(350, 200, image=img, anchor="center")

    # button to close the tkinter window
    exit_button = Button(master, text="Submit", height=10, width=15, command=master.destroy)
    exit_button.pack(pady=80)
 
    mainloop()

    return v.get()

def save_to_folder(style, img, file):
    style_to_folder = {"1": "Kufi", "2":"Farisi", "3":"Thuluth", "4":"Diwani", "5":"Other"}
    folder = style_to_folder[style]
    img.save(PATH+folder+"/"+file)


def main():

    
    files = os.listdir(PATH)
    files = [file for file in files if os.path.isfile(PATH+file)] # ignore the folders

    for file in files:

        if file == '.DS_Store': # if the file is meta file, ignore it
            continue

        img = PIL.Image.open(PATH + file)
        style = get_calligraphy_style(PATH + file)
        # img.save(save_to_path + filename + '.jpg')
        save_to_folder(style, img, file)
        os.remove(PATH+file)


if __name__ == "__main__":
    main()