from gan import datasets


def test_mnist_dataset():
    """
    Check if the dataset is well formed
    """
    mnist = datasets.fetch_mnist()
    assert len(mnist.data) == 70000
    assert mnist.data[0].shape == (32, 32)
