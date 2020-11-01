import torch
from torch import nn
from src.nn.stylegan.utils import NoiseInjection, AdaIN, Mapping


class GeneratorBlock(nn.Module):
    """
    StyleGAN Generator Block Class
    """

    def __init__(self, input_channels, output_channels, noise_dim, kernel_size, starting_size, use_upsample=True,
                 alpha=0.1):

        super().__init__()
        self.use_upsample = use_upsample
        if self.use_upsample:
            self.upsample = nn.Upsample((starting_size, starting_size), mode='bilinear')
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, padding=1)
        self.inject_noise = NoiseInjection(output_channels)
        self.adain = AdaIN(output_channels, noise_dim)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, generator_input, noise_vector):
        '''
        Function for completing a forward pass of MicroStyleGANGeneratorBlock: Given an x and w,
        computes a StyleGAN generator block.
        Parameters:
            x: the input into the generator, feature map of shape (n_samples, channels, width, height)
            w: the intermediate noise vector
        '''
        if self.use_upsample:
            generator_input = self.upsample(generator_input)
        return self.adain(
            self.activation(
                self.inject_noise(
                    self.conv(generator_input)
                )
            ), noise_vector
        )


class Generator(nn.Module):
    def __init__(self, input_noise_dim, hidden_noise_dim, output_noise_dim, input_channels, hidden_channels,
                 outpput_channels, kernel_size):
        super().__init__()

        self.map = Mapping(input_noise_dim, hidden_noise_dim, output_noise_dim)

        self.starting_constant = torch.ones((1, input_channels, 4, 4))

    # TODO: finish