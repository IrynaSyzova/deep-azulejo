import torch
from torch import nn


class Mapping(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        MLP of 3 layers
        :param z_dim:
        :param w_dim:
        """
        super().__init__()
        self.mapping = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, noise):
        return self.mapping(noise)


class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = torch.randn((1, channels, 1, 1))

    def forward(self, image):
        """
        Given an image, returns image with random noise
        """
        noise = torch.randn((image.shape[0], 1, image.shape[2], image.shape[3]), device=image.device)

        return image + self.weight(noise)


class AdaIN(nn.Module):
    """
    Adaptive instance normalisation
    """

    def __init__(self, channels, noise_vector_dim):
        super().__init__()

        self.instance_norm = nn.InstanceNorm2d(channels)

        self.style_scale = nn.Linear(noise_vector_dim, channels)
        self.style_shift = nn.Linear(noise_vector_dim, channels)

    def forward(self, image, noise_vector):
        """
        Given an image and a noise vector, returns image, that was normalised, scaled, and shifted
        """
        image_norm = self.instance_norm(image)
        style_scale = self.style_scale_transform(noise_vector)[:, :, None, None]
        style_shift = self.style_shift_transform(noise_vector)[:, :, None, None]
        return image_norm * self.style_scale + self.style_shift
