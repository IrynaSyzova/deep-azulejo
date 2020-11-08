import torch
from torch import nn


class Critic(nn.Module):
    def __init__(self, channels=3, hidden_dim=64):
        """
        Critic class
        :param channels: number of channels in the images
        """
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            self._make_block(channels, hidden_dim, kernel_size=3, stride=1),
            self._make_block(hidden_dim, hidden_dim*2, kernel_size=3, stride=2),
            self._make_block(hidden_dim*2, hidden_dim *4, kernel_size=3, stride=2),
            self._make_final_block(hidden_dim*4, 1, kernel_size=4, stride=2)
        )

    @staticmethod
    def _make_block(input_channels, output_channels, kernel_size=4, stride=2, alpha=0.2, padding=1):
        """
        Building block for Critic, which is convolution-batch norm-leaky relu combination
        :param input_channels: input channels size
        :param output_channels: output channel size
        :param kernel_size: filter size of the convolution
        :param stride: stride of the convolution
        :param alpha: Leaky ReLy parameter
        """
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding=padding),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(alpha)
        )

    @staticmethod
    def _make_final_block(input_channels, output_channels, kernel_size=4, stride=2, padding=1):
        """
        Final block for Critic, which is just a convolution
        :param input_channels: input channels size
        :param output_channels: output channel size
        :param kernel_size: filter size of the convolution
        :param stride: stride of the convolution
        """
        return nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding=padding))
        )

    def forward(self, image):
        """
        Forward pass of Critic
        :param image: a flattened image tensor
        """
        result = self.critic(image)
        return result.view(len(result), -1)

    def loss(self, fake_imgs, real_imgs, penalty_weight, device='cpu'):
        """
        Loss for critic given predictions on generated and real images
        :param fake_imgs: generated images
        :param real_imgs: real images
        :param penalty_weight: weight of gradient penalty
        """
        critic_fake_pred = self.critic(fake_imgs)
        critic_real_pred = self.critic(real_imgs)
        epsilon = torch.rand(len(real_imgs), 1, 1, 1, device=device, requires_grad=True)
        gradient = self.get_gradient(fake_imgs, real_imgs, epsilon)
        gradient_penalty = self.get_gradient_penalty(gradient)
        return critic_fake_pred.mean() - critic_real_pred.mean() + gradient_penalty * penalty_weight

    def get_gradient(self, fake_imgs, real_imgs, epsilon):
        """
        Gradient of critic's scores for a mix of real and fake images
        :param critic: critic
        :param real_imgs: batch of real images
        :param fake_imgs: batch of fake images
        :param epsilon: a parameter to create a fake image from real and fake image
        """
        mixed_imgs = real_imgs * epsilon + fake_imgs * (1 - epsilon)
        mixed_scores = self.critic(mixed_imgs)

        return torch.autograd.grad(
            inputs=mixed_imgs,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True
        )[0]

    @staticmethod
    def get_gradient_penalty(gradient):
        """
        Gradient's penalty, which is it's L2 norm
        :param gradient: gradient we are penalising
        """
        gradient = gradient.view(len(gradient), -1)

        gradient_norm = gradient.norm(2, dim=1)

        return sum((gradient_norm - 1) ** 2) / len(gradient_norm)
