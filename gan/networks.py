import math
import torch
import torch.nn as nn


def get_nb_layers_from_res(resolution):
    """
    Return the number of intermediate layers of the generator
    """
    assert resolution in [8, 16, 32, 64, 128]
    return int(math.log(resolution, 2) - 1)


def weights_init(m):
    """
    Initialize the weight of the different layers
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    """
    Generator mostly following the implementation of the DCGAN (https://arxiv.org/abs/1511.06434).
    """

    def __init__(self, latent_dim=16, img_channels=1, img_res=32, feature_map_size=32):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.img_channels = img_channels
        self.img_res = img_res
        self.nb_layers = get_nb_layers_from_res(img_res)
        self.feature_map_size = feature_map_size

        layers = []

        out_channels = int(
            self.feature_map_size * img_res / 4
        )  # The last layer takes a 4*4 square

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

        self.generator = nn.Sequential(*layers, nn.Tanh()).apply(weights_init)

    def get_noise(self, device, batch_size=1):
        return torch.randn(batch_size, self.latent_dim, 1, 1, device=device)

    def forward(self, z):
        return self.generator(z)


class Discriminator(nn.Module):
    def __init__(self, img_channels=1, img_res=32, feature_map_size=16):
        super(Discriminator, self).__init__()

        self.img_channels = img_channels
        self.img_res = img_res
        self.nb_layers = get_nb_layers_from_res(img_res)
        self.feature_map_size = feature_map_size

        layers = [
            nn.Conv2d(
                self.img_channels,
                feature_map_size,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        in_channels = feature_map_size

        for i in range(self.nb_layers - 2):

            out_channels = int(in_channels * 2)

            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

            in_channels = out_channels

        layers.append(
            nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=0, bias=False)
        )

        self.discriminator = nn.Sequential(*layers, nn.Sigmoid()).apply(weights_init)

    def forward(self, x):
        return self.discriminator(x).view(-1)
