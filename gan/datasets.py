import numpy as np
from sklearn.datasets import fetch_openml


def fetch_mnist():
    """
    Fetch the mnist dataset from openml
    - The images are padded to be of shape 32*32
    - The pixel are rescaled to be between -1 and 1 (for the tanh)
    """
    mnist = fetch_openml('mnist_784', data_home=".")
    
    rescale = lambda x : x/255*2 -1
    mnist.data = [rescale(np.pad(i_data.reshape((28,28)), 2)) for i_data, i_target in zip(mnist.data, mnist.target)]
    mnist.target = mnist.target.astype(int)
    
    return mnist