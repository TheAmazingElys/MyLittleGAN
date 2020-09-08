import numpy as np, matplotlib.pyplot as plt

"""
Related to the creation of the grid of images
"""

def reshape_gray(imgs, img_size):
    return [i_img.reshape(img_size, img_size) for i_img in imgs]

def reshape_rgb(imgs, img_size):
    return [i_img.reshape(3, img_size, img_size).transpose(0,1).transpose(1,2)/2+0.5 for i_img in imgs]

def make_grid(imgs, img_size=32, img_per_row=8, cmap = "gray"):
    """
    TODO infer img_size and channels from imgs to remove img_size and cmap
    """
    img_per_row = min(len(imgs), img_per_row)
    imgs = reshape_gray(imgs, img_size) if cmap == "gray" else reshape_rgb(imgs, img_size)
    n_rows = (len(imgs) - 1) // img_per_row + 1
    rows_of_images = []
    n_empty = n_rows * img_per_row - len(imgs)
    imgs.append(np.zeros((img_size, img_size * n_empty)))

    for i_row in range(n_rows):
        i_imgs = imgs[i_row * img_per_row : (i_row + 1) * img_per_row]
        rows_of_images.append(np.concatenate(i_imgs, axis=1))

    return np.concatenate(rows_of_images, axis=0)


def plot_matrix(matrix, cmap="gray", axis="off"):
    
    if cmap=="gray":
        plt.imshow(matrix, cmap=cmap)
    else:
        plt.imshow(matrix)
        
    plt.axis(axis)
    return plt


def plot_img(imgs, img_size=32, img_per_row=8, cmap = "gray", file_name=None):
    if file_name:
        assert file_name[-4:] == ".jpg"

    matrix = make_grid(imgs, img_size = img_size, img_per_row = img_per_row, cmap = cmap)
    plt = plot_matrix(matrix, cmap = cmap)

    if file_name:
        plt.savefig(file_name, bbox_inches="tight", pad_inches=0.05)


"""
Signals are all you need
"""


class Signal:
    def __init__(self):
        self.connected = []

    def connect(self, function):
        """ The connected functions will be called when the signal will emit"""
        if function not in self.connected:
            self.connected.append(function)

    def emit(self, *args, **kwargs):
        """ Call the connected functions """
        for i_function in self.connected:
            i_function(*args, **kwargs)
