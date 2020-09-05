import math
import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    Generator mostly following the implementation of the DCGAN.
    """

    def __init__(self, latent_dim=16, img_channels=1, nb_layers=4, feature_map_size=32):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.img_channels = img_channels
        self.nb_layers = nb_layers
        self.feature_map_size = feature_map_size

        layers = []

        out_channels = self.feature_map_size * int(
            math.pow(2, self.nb_layers - 1)
        )

        layers.append(
            nn.ConvTranspose2d(
                in_channels=self.latent_dim,
                out_channels=out_channels,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            )
        )

        """
        Strides of 2 on the ConvTranspose2d with knernel_size of 4 and padding of 1 will double the resolution of the output at each layer.
        Each layer after the first double de resolution.
        
        Notice that BatchNorm2D and ReLU are applied after each ConvTranspose2D intended to be fed to another (so not the last one!)
        """

        for i_layer in reversed(range(self.nb_layers - 1)):

            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(out_channels))

            in_channels = out_channels
            out_channels = int(out_channels / 2)
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels
                    if i_layer != 0
                    else self.img_channels,  # The last layer should have the same number of channels than the generated image
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )

        self.generator = nn.Sequential(*layers, nn.Tanh())

    def get_noise(self, device):
        return torch.randn(1, self.latent_dim, 1, 1, device=device)

    def forward(self, z):
        return self.generator(z)
