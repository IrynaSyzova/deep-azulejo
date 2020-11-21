import torch
from torch import nn
from scipy.stats import truncnorm


class Generator(nn.Module):
    """
    Generator class

    :param z_dim: input noise dimension
    :param channels: number of channels in the images
    :param n_features: controls width of convolutional layers
    """
    def __init__(self, z_dim=100, channels=3, n_features=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.channels = channels

        self.generator = nn.Sequential(
            self.__get_block(z_dim, n_features * 8, kernel_size=4, stride=1, padding=0),
            self.__get_block(n_features * 8, n_features * 4, kernel_size=4, stride=2, padding=1),
            self.__get_block(n_features * 4, n_features * 2, kernel_size=4, stride=2, padding=1),
            self.__get_block(n_features * 2, n_features, kernel_size=4, stride=2, padding=1),
            self.__get_block(n_features, n_features, kernel_size=3, stride=1, padding=1),
            self.__get_final_block(n_features, channels, kernel_size=4, stride=2, padding=1)
        )

    @staticmethod
    def __get_block(input_channels, output_channels, kernel_size=5, stride=1, padding=1):
        """
        Building block for Generator, which is transposed convolution-batch norm-relu combination
        :param input_channels: input channels size
        :param output_channels: output channel size
        :param kernel_size: filter size of the convolution
        :param stride: stride of the convolution
        :param padding: padding of the convolution
        """
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding=padding, output_padding=0,
                               dilation=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def __get_final_block(input_channels, output_channels, kernel_size=5, stride=1, padding=0):
        """
        Final block for Generator, which is transposed convolution-tanh combination
        :param input_channels: input channels size
        :param output_channels: output channel size
        :param kernel_size: filter size of the convolution
        :param stride: stride of the convolution
        :param padding: padding of the convolution
        """
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding=padding, output_padding=0,
                               dilation=1),
            nn.Tanh()
        )

    def forward(self, noise):
        """
        Forward pass of Generator
        :param noise: input noise
        """
        x = noise.view(len(noise), self.z_dim, 1, 1)
        return self.generator(x)

    def get_noise(self, n_samples, device='cpu'):
        """
        Noise vector for Generator
        :param n_samples: number of samples to generate
        :param device: device type
        """
        return torch.randn(n_samples, self.z_dim, device=device)

    def get_truncated_noise(self, n_samples, truncation):
        """
        Noise vector for Generator of truncated normal distribution
        :param n_samples: number of samples to generate
        :param z_dim: noise dimension
        :param truncation: truncation value, the noise will be truncated from -truncation to truncation
        """
        return torch.Tensor(truncnorm.rvs(-truncation, truncation, size=(n_samples, self.z_dim)))
