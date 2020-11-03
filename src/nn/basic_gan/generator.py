import torch
from torch import nn
from scipy.stats import truncnorm


class Generator(nn.Module):
    def __init__(self, z_dim=100, channels=3, hidden_dim=64):
        """
        Generator class

        :param z_dim: input noise dimension
        :param channels: number of channels in the images
        """
        super().__init__()
        self.z_dim = z_dim
        self.channels = channels

        self.generator = nn.Sequential(
            self._make_block(z_dim, hidden_dim*4, kernel_size=3, stride=2),
            self._make_block(hidden_dim*4, hidden_dim*2, kernel_size=4, stride=1),
            self._make_final_block(hidden_dim*2, channels, kernel_size=4, stride=2  )
        )

    @staticmethod
    def _make_block(input_channels, output_channels, kernel_size=5, stride=1):
        """
        Building block for Generator, which is transposed convolution-batch norm-relu combination
        :param input_channels: input channels size
        :param output_channels: output channel size
        :param kernel_size: filter size of the convolution
        :param stride: stride of the convolution
        """
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def _make_final_block(input_channels, output_channels, kernel_size=5, stride=1):
        """
        Final block for Generator, which is transposed convolution-tanh combination
        :param input_channels: input channels size
        :param output_channels: output channel size
        :param kernel_size: filter size of the convolution
        :param stride: stride of the convolution
        """
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
            nn.Tanh()
        )

    def forward(self, noise):
        """
        Forward pass of Generator
        :param noise: input noise
        """
        x = noise.view(len(noise), self.z_dim, 1, 1)
        return self.generator(x)

    @staticmethod
    def get_noise(n_samples, z_dim, device='cpu'):
        """
        Noise vector for Generator
        :param n_samples: number of samples to generate
        :param z_dim: noise dimension
        :param device: device type
        """
        return torch.randn(n_samples, z_dim, device=device)

    @staticmethod
    def get_truncated_noise(n_samples, z_dim, truncation):
        """
        Noise vector for Generator of truncated normal distribution
        :param n_samples: number of samples to generate
        :param z_dim: noise dimension
        :param truncation: truncation value, the noise will be truncated from -truncation to truncation
        """
        return torch.Tensor(truncnorm.rvs(-truncation, truncation, size=(n_samples, z_dim)))

    @staticmethod
    def loss(critic_fake_pred):
        """
        Generator's loss given critic's scores for generated images
        :param critic_fake_pred: scores for generated images
        """
        return critic_fake_pred.mean() * (-1)
