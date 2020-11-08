import torch
from torch import nn
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

from fastprogress.fastprogress import master_bar, progress_bar

from src.nn.utils import plot_batch


class GAN:
    def __init__(self, generator, critic, optimiser, device='cpu'):
        self.device = device
        self.generator = generator.to(device)
        self.critic = critic.to(device)
        self.generator_optimiser = optimiser(generator.parameters())
        self.critic_optimiser = optimiser(critic.parameters())

    def train(self, data_loader, n_epochs, init=True,
              checkpoint_folder=None,
              critic_repeats=5, gradient_penalty_weight=10):
        noise_dimension = self.generator.z_dim

        if checkpoint_folder is None:
            checkpoint_folder = "models/basic_gan_{}".format(str(datetime.today().date()))
            Path(checkpoint_folder).mkdir(parents=True, exist_ok=True)

        if init:
            self.generator.apply(init_weights)
            self.critic.apply(init_weights)

        generator_losses, critic_losses = [], []

        master_progress_bar = master_bar(range(n_epochs))
        master_progress_bar.names = ['generator', 'critic']
        for epoch in master_progress_bar:
            generator_losses_epoch, critic_losses_epoch = [], []
            for real_imgs in progress_bar(data_loader, parent=master_progress_bar):
                batch_size = len(real_imgs)
                real_imgs = real_imgs.to(self.device)

                mean_critic_loss = 0
                for _ in range(critic_repeats):
                    self.critic_optimiser.zero_grad()

                    fake_imgs = self.generator(
                        self.generator.get_noise(batch_size, noise_dimension, device=self.device))

                    critic_loss = self.critic.loss(fake_imgs, real_imgs, gradient_penalty_weight, device=self.device)

                    mean_critic_loss += critic_loss.item() / critic_repeats

                    critic_loss.backward(retain_graph=True)
                    # Update optimizer
                    self.critic_optimiser.step()

                critic_losses_epoch.append(mean_critic_loss)

                self.generator_optimiser.zero_grad()

                fake_imgs = self.generator(self.generator.get_noise(batch_size, noise_dimension, device=self.device))
                fake_pred = self.critic(fake_imgs)

                generator_loss = self.generator.loss(fake_pred)
                generator_loss.backward()
                self.generator_optimiser.step()
                generator_losses_epoch.append(generator_loss.item())

            print("Epoch {epoch}: Generator loss: {gen_mean}, critic loss: {crit_mean}".format(
                epoch=epoch,
                gen_mean=np.mean(generator_losses_epoch),
                crit_mean=np.mean(critic_losses_epoch)
            ))

            plot_batch(fake_imgs, device=self.device, caption='Generated images')
            plot_batch(real_imgs, device=self.device, caption='Real images')

            # self.update_progress_plot(epoch, n_epochs, generator_losses, critic_losses, master_progress_bar)

            generator_checkpoint_path = '{}/generator_{}.pt'.format(checkpoint_folder, epoch)
            critic_checkpoint_path = '{}/critic_{}.pt'.format(checkpoint_folder, epoch)

            save_checkpoint(self.generator, self.generator_optimiser, generator_checkpoint_path, epoch,
                            np.mean(generator_losses_epoch))
            save_checkpoint(self.critic, self.critic_optimiser, critic_checkpoint_path, epoch,
                            np.mean(critic_losses_epoch))

            generator_losses.append(np.mean(generator_losses_epoch))
            critic_losses.append(np.mean(critic_losses_epoch))

        self.plot_progress_plot(n_epochs, generator_losses, critic_losses)

    @staticmethod
    def plot_progress_plot(epochs, generator_losses, critic_losses):
        x = range(1, len(epochs) + 1)
        plt.plot(x, generator_losses, label='generator')
        plt.plot(x, critic_losses, label='critic')
        plt.show()


def init_weights(layer, std=0.01):
    """
    Weights initiation for networks
    :param layer: layer for which we initiate the weights
    :param std: weight's std
    """
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d) or isinstance(layer, nn.BatchNorm2d):
        nn.init.normal_(layer.weight, 0.0, std)
    if isinstance(layer, nn.BatchNorm2d):
        nn.init.constant_(layer.bias, 0)


def save_checkpoint(net, optimiser, path, epoch, loss):
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
        'loss': loss,
    }, path)
