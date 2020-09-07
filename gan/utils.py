import numpy as np, matplotlib.pyplot as plt

def make_grid(imgs, img_size = 32, img_per_row = 8):
    """
    TODO infer img_size from imgs
    """
    img_per_row = min(len(imgs), img_per_row)
    imgs = [i_img.reshape(img_size,img_size) for i_img in imgs]
    n_rows = (len(imgs) - 1) // img_per_row + 1
    rows_of_images = []
    n_empty = n_rows * img_per_row - len(imgs)
    imgs.append(np.zeros((img_size, img_size * n_empty)))
    
    for i_row in range(n_rows):
        i_imgs = imgs[i_row * img_per_row : (i_row +1) * img_per_row]
        rows_of_images.append(np.concatenate(i_imgs, axis=1))
        
    return np.concatenate(rows_of_images, axis = 0)

def plot_matrix(matrix, cmap = "gray", axis = "off"):
    plt.imshow(matrix, cmap = cmap)
    plt.axis(axis)
    
    return plt
    
def plot_img(imgs, img_size = 32, img_per_row = 8, file_name = None):
    if file_name:
        assert file_name[-4:] == ".jpg"
    
    matrix = make_grid(imgs)
    plt = plot_matrix(matrix)
    
    if file_name:
        plt.savefig(file_name, bbox_inches = 'tight', pad_inches = 0.05)