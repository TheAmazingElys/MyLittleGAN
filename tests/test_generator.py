from gan import networks
import numpy as np


def test_generator_output_shape():
    """
    Check if the output of the generator is well formed
    """

    gen = networks.Generator(
        latent_dim=16, img_channels=1, nb_layers=4, feature_map_size=32
    )
    assert (gen(gen.get_noise("cpu")).shape == np.array([1, 1, 32, 32])).all()
