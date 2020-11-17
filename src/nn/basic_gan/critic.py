import torch
from torch import nn


class Critic(nn.Module):
    def __init__(self, channels=3, n_features=64, spectral_normalisation=False, batch_normalisation=True):
        """
        Critic class
        :param channels: number of channels in the images
        """
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            self.__get_block(channels, n_features, kernel_size=4, stride=2,
                             spectral_normalisation=spectral_normalisation,
                             batch_normalisation=False),
            self.__get_block(n_features, n_features, kernel_size=3, stride=1, padding=0,
                             spectral_normalisation=spectral_normalisation,
                             batch_normalisation=batch_normalisation),
            self.__get_block(n_features, n_features * 2, kernel_size=4, stride=2,
                             spectral_normalisation=spectral_normalisation,
                             batch_normalisation=batch_normalisation),
            self.__get_block(n_features * 2, n_features * 4, kernel_size=3, stride=2,
                             spectral_normalisation=spectral_normalisation,
                             batch_normalisation=batch_normalisation),
            self.__get_block(n_features * 4, n_features * 8, kernel_size=3, stride=2,
                             spectral_normalisation=spectral_normalisation,
                             batch_normalisation=batch_normalisation),
            self.__get_final_block(n_features * 8, 1, kernel_size=4, stride=1, padding=0,
                                   spectral_normalisation=spectral_normalisation)
        )

    @staticmethod
    def __get_block(input_channels, output_channels, kernel_size=4, stride=2, alpha=0.2, padding=1,
                    spectral_normalisation=False, batch_normalisation=True):
        """
        Building block for Critic, which is convolution-batch norm-leaky relu combination
        :param input_channels: input channels size
        :param output_channels: output channel size
        :param kernel_size: filter size of the convolution
        :param stride: stride of the convolution
        :param alpha: Leaky ReLy parameter
        """
        conv_layer = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding=padding)
        if spectral_normalisation:
            conv_layer = nn.utils.spectral_norm(conv_layer)
        layers = [conv_layer]
        if batch_normalisation:
            layers += [nn.BatchNorm2d(output_channels)]
        layers += [nn.LeakyReLU(alpha, inplace=True)]
        return nn.Sequential(*layers)

    @staticmethod
    def __get_final_block(input_channels, output_channels, kernel_size=4, stride=2, padding=1,
                          spectral_normalisation=False):
        """
        Final block for Critic, which is just a convolution
        :param input_channels: input channels size
        :param output_channels: output channel size
        :param kernel_size: filter size of the convolution
        :param stride: stride of the convolution
        """
        layer = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding=padding)
        if spectral_normalisation:
            layer = nn.utils.spectral_norm(layer)
        return nn.Sequential(layer)

    def forward(self, image):
        """
        Forward pass of Critic
        :param image: a flattened image tensor
        """
        result = self.critic(image)
        return result.view(len(result), -1)
