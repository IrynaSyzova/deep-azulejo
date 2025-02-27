import torch
from torch import nn
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

from fastprogress.fastprogress import master_bar, progress_bar

from src.nn.utils import plot_batch, init_weights, save_checkpoint


class GAN:
    """
    Basic Generative Adversarial Network.

    Attributes:
        generator: A neural network describing generator part of GAN
        critic: A neural network describing critic (or discriminator) part of GAN
        generator_optimiser: Optimiser for generator
        critic_optimiser: Optimiser for critic
        device: 'cpu' or 'cuda:0'
        checkpoint_folder: Where to dave checkpoints; if None, it will be models/basic_gan_<current_date>
        init: whether to init generator and critic, default True.
            Set to False if you want to extend previously trained netrworks.
        critic repeats: Number of critic optimisation steps to 1 generator step
    """
    def __init__(self, generator, critic, generator_optimiser, critic_optimiser, device='cpu', checkpoint_folder=None,
                 init=True, critic_repeats=1):
        self.device = device
        self.generator = generator.to(device)
        self.critic = critic.to(device)
        self.generator_optimiser = generator_optimiser(generator.parameters())
        self.critic_optimiser = critic_optimiser(critic.parameters())
        self.critic_repeats = critic_repeats
        self.__criterion = nn.BCEWithLogitsLoss()

        if checkpoint_folder is None:
            checkpoint_folder = "models/basic_gan_{}".format(str(datetime.today().date()))
        self.checkpoint_folder = checkpoint_folder
        Path(self.checkpoint_folder).mkdir(parents=True, exist_ok=True)

        if init:
            self.generator.apply(init_weights)
            self.critic.apply(init_weights)

    def train(self, data_loader, n_epochs, show_imgs_number=32):
        """
        Trains generator and critic of GAN.
        :param data_loader: Source of data
        :param n_epochs: Number of epochs to train
        :param show_imgs_number: How many images to print after each epoch
        """
        generator_losses, critic_losses = [], []

        master_progress_bar = master_bar(range(n_epochs))
        master_progress_bar.names = ['generator', 'critic']
        for epoch in master_progress_bar:
            generator_losses_epoch, critic_losses_epoch = [], []
            for real_imgs in progress_bar(data_loader, parent=master_progress_bar):
                critic_loss, generator_loss = self.__step(real_imgs)

                critic_losses_epoch.append(critic_loss)
                generator_losses_epoch.append(generator_loss)

            print("Epoch {epoch}: Generator loss: {gen_last}, critic loss: {crit_last}".format(
                epoch=epoch,
                gen_last=generator_losses_epoch[-1],
                crit_last=critic_losses_epoch[-1]
            ))

            fake_imgs = self.generator(
                self.generator.get_noise(show_imgs_number, device=self.device))

            plot_batch(fake_imgs, plot_size=show_imgs_number, caption='Generated images')

            # self.update_progress_plot(epoch, n_epochs, generator_losses, critic_losses, master_progress_bar)

            generator_checkpoint_path = '{}/generator_{}.pt'.format(self.checkpoint_folder, epoch)
            critic_checkpoint_path = '{}/critic_{}.pt'.format(self.checkpoint_folder, epoch)

            save_checkpoint(self.generator, self.generator_optimiser, generator_checkpoint_path, epoch,
                            generator_losses_epoch[-1])
            save_checkpoint(self.critic, self.critic_optimiser, critic_checkpoint_path, epoch,
                            critic_losses_epoch[-1])

            generator_losses.extend(generator_losses_epoch)
            critic_losses.extend(critic_losses_epoch)

        self.plot_progress_plot(generator_losses, critic_losses)

    def __step(self, real_imgs):
        """
        Training step
        :param real_imgs: Batch of real images to use for optimisation step
        """
        batch_size = len(real_imgs)
        real_imgs = real_imgs.to(self.device)

        mean_critic_loss = 0
        for _ in range(self.critic_repeats):
            fake_imgs = self.generator(self.generator.get_noise(batch_size, device=self.device)).detach()
            critic_loss = self.__critic_optimiser_step(fake_imgs, real_imgs)
            mean_critic_loss += critic_loss / self.critic_repeats

        fake_imgs = self.generator(self.generator.get_noise(batch_size, device=self.device))
        generator_loss = self.__generator_optimiser_step(fake_imgs)

        return mean_critic_loss, generator_loss

    def __critic_optimiser_step(self, fake_imgs, real_imgs):
        self.critic_optimiser.zero_grad()
        critic_loss = self._critic_loss(fake_imgs, real_imgs)
        critic_loss.backward(retain_graph=True)
        self.critic_optimiser.step()
        return critic_loss.item()

    def __generator_optimiser_step(self, fake_imgs):
        self.generator_optimiser.zero_grad()
        generator_loss = self._generator_loss(fake_imgs)
        generator_loss.backward()
        self.generator_optimiser.step()

        return generator_loss.item()

    def _critic_loss(self, fake_imgs, real_imgs):
        critic_fake_pred = self.critic(fake_imgs)
        critic_fake_loss = self.__criterion(critic_fake_pred, torch.zeros_like(critic_fake_pred))
        critic_real_pred = self.critic(real_imgs)
        critic_real_loss = self.__criterion(critic_real_pred, torch.ones_like(critic_real_pred))
        return (critic_fake_loss + critic_real_loss) / 2.

    def _generator_loss(self, fake_imgs):
        critic_fake_pred = self.critic(fake_imgs)
        return self.__criterion(critic_fake_pred, torch.ones_like(critic_fake_pred))

    @staticmethod
    def plot_progress_plot(generator_losses, critic_losses):
        """
        Plots generator and critic loss progress over epochs
        """
        fig, ax = plt.subplots(2, 1, figsize=(12, 7))
        x = range(1, len(generator_losses) + 1)
        ax[0].plot(x, generator_losses, label='generator')
        ax[0].set_title('Generator losses')
        ax[1].plot(x, critic_losses, label='critic')
        ax[1].set_title('Critic losses')
        plt.show()


class WGAN(GAN):
    """
    Wasserstein Generative Adversarial Network.

    Attributes:
        gradient_penalty_weight: weight of gradient penalty in critic loss
        checkpoint_folder: Where to dave checkpoints; if None, it will be models/wgan_<current_date>

    """
    def __init__(self, generator, critic, generator_optimiser, critic_optimiser, device='cpu',
                 gradient_penalty_weight=0, critic_repeats=5, checkpoint_folder=None, init=True):
        if checkpoint_folder is None:
            checkpoint_folder = "models/wgan_{}".format(str(datetime.today().date()))
        super(WGAN, self).__init__(generator, critic, generator_optimiser, critic_optimiser, device=device,
                                   checkpoint_folder=checkpoint_folder, init=init, critic_repeats=critic_repeats)
        self.gradient_penalty_weight = gradient_penalty_weight

    def _generator_loss(self, fake_imgs):
        critic_fake_pred = self.critic(fake_imgs)
        return critic_fake_pred.mean() * (-1)

    def _critic_loss(self, fake_imgs, real_imgs):
        critic_fake_pred = self.critic(fake_imgs)
        critic_real_pred = self.critic(real_imgs)
        gradient = self.__critic_loss_gradient(fake_imgs, real_imgs)
        gradient_penalty = self.__get_gradient_penalty(gradient)
        return critic_fake_pred.mean() - critic_real_pred.mean() + gradient_penalty * self.gradient_penalty_weight

    def __critic_loss_gradient(self, fake_imgs, real_imgs):
        """
        Gradient of critic's scores for a mix of real and fake images
        :param critic: critic
        :param real_imgs: batch of real images
        :param fake_imgs: batch of fake images
        :param epsilon: a parameter to create a fake image from real and fake image
        """
        epsilon = torch.rand(len(real_imgs), 1, 1, 1, device=self.device, requires_grad=True)
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
    def __get_gradient_penalty(gradient):
        """
        Gradient's penalty, which is it's L2 norm
        :param gradient: gradient we are penalising
        """
        gradient = gradient.view(len(gradient), -1)

        gradient_norm = gradient.norm(2, dim=1)

        return sum((gradient_norm - 1) ** 2) / len(gradient_norm)
