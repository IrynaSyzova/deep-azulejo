import torch
from torch import nn


class Critic(nn.Module):
    def __init__(self, channels=3, hidden_dim=64, spectral_normalisation=False):
        """
        Critic class
        :param channels: number of channels in the images
        """
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            self._make_block(channels, hidden_dim, kernel_size=3, stride=1,
                             spectral_normalisation=spectral_normalisation),
            self._make_block(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2,
                             spectral_normalisation=spectral_normalisation),
            self._make_block(hidden_dim * 2, hidden_dim * 4, kernel_size=3, stride=2,
                             spectral_normalisation=spectral_normalisation),
            self._make_final_block(hidden_dim * 4, 1, kernel_size=4, stride=2,
                                   spectral_normalisation=spectral_normalisation)
        )

    @staticmethod
    def _make_block(input_channels, output_channels, kernel_size=4, stride=2, alpha=0.2, padding=1,
                    spectral_normalisation=False):
        """
        Building block for Critic, which is convolution-batch norm-leaky relu combination
        :param input_channels: input channels size
        :param output_channels: output channel size
        :param kernel_size: filter size of the convolution
        :param stride: stride of the convolution
        :param alpha: Leaky ReLy parameter
        """
        if spectral_normalisation:
            return nn.Sequential(
                nn.utils.spectral_norm(
                    nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding=padding)),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(alpha)
            )
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding=padding),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(alpha)
        )

    @staticmethod
    def _make_final_block(input_channels, output_channels, kernel_size=4, stride=2, padding=1,
                          spectral_normalisation=False):
        """
        Final block for Critic, which is just a convolution
        :param input_channels: input channels size
        :param output_channels: output channel size
        :param kernel_size: filter size of the convolution
        :param stride: stride of the convolution
        """
        if spectral_normalisation:
            return nn.Sequential(
                nn.utils.spectral_norm(
                    nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding=padding)
                )
            )
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding=padding)
        )

    def forward(self, image):
        """
        Forward pass of Critic
        :param image: a flattened image tensor
        """
        result = self.critic(image)
        return result.view(len(result), -1)
