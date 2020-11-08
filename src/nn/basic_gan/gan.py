import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime

from fastprogress.fastprogress import master_bar, progress_bar

from src.nn.basic_gan.generator import Generator
from src.nn.basic_gan.critic import Critic
from src.nn.utils import plot_batch


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


def update_progress_plot(epoch, epochs, generator_losses, critic_losses, master_bar):
    x = range(1, epoch + 1)
    generator_loss = torch.Tensor(generator_losses).view(-1, len(generator_losses))
    critic_loss = torch.Tensor(critic_losses).view(-1, len(critic_losses))
    y = np.concatenate((generator_loss, critic_loss))
    graphs = [[x, generator_loss], [x, critic_loss]]
    x_margin = 0.2
    y_margin = 0.05
    x_bounds = [1 - x_margin, epochs + x_margin]
    y_bounds = [np.min(y) - y_margin, np.max(y) + y_margin]

    master_bar.update_graph(graphs, x_bounds, y_bounds)


def train(data_loader, noise_dimension, n_epochs,
          device='cpu',
          checkpoint_folder=None,
          critic_repeats=5, lr=0.0001, betas=(0.9, 0.999), gradient_penalty_weight=10):
    if checkpoint_folder is None:
        checkpoint_folder = "models/basic_gan_{}".format(str(datetime.today().date()))
        Path(checkpoint_folder).mkdir(parents=True, exist_ok=True)

    generator = Generator(noise_dimension).to(device)
    critic = Critic().to(device)

    generator.apply(init_weights)
    critic.apply(init_weights)

    generator_optimiser = torch.optim.Adam(generator.parameters(), lr=lr, betas=betas)
    critic_optimiser = torch.optim.Adam(critic.parameters(), lr=lr, betas=betas)

    generator_losses, critic_losses = [], []

    master_progress_bar = master_bar(range(n_epochs))
    master_progress_bar.names = ['generator', 'critic']
    for epoch in master_progress_bar:
        for real_imgs in progress_bar(data_loader, parent = master_progress_bar):
            batch_size = len(real_imgs)
            real_imgs = real_imgs.to(device)

            mean_critic_loss = 0
            for _ in range(critic_repeats):
                critic_optimiser.zero_grad()

                fake_imgs = generator(generator.get_noise(batch_size, noise_dimension, device=device))

                critic_loss = critic.loss(fake_imgs, real_imgs, gradient_penalty_weight, device=device)

                mean_critic_loss += critic_loss.item() / critic_repeats

                critic_loss.backward(retain_graph=True)
                # Update optimizer
                critic_optimiser.step()

            critic_losses.append(mean_critic_loss)

            generator_optimiser.zero_grad()

            fake_imgs = generator(generator.get_noise(batch_size, noise_dimension, device=device))
            fake_pred = critic(fake_imgs)

            generator_loss = generator.loss(fake_pred)
            generator_loss.backward()
            generator_optimiser.step()
            generator_losses.append(generator_loss.item())

        print("Epoch {epoch}: Generator loss: {gen_mean}, critic loss: {crit_mean}".format(
            epoch=epoch,
            gen_mean=np.mean(generator_losses),
            crit_mean=np.mean(critic_losses)
        ))

        plot_batch(fake_imgs, device=device, caption='Generated images')
        plot_batch(real_imgs, device=device, caption='Real images')

        update_progress_plot(epoch, n_epochs, generator_losses, critic_losses, master_progress_bar)

        generator_checkpoint_path = '{}/generator_{}.pt'.format(checkpoint_folder, epoch)
        critic_checkpoint_path = '{}/critic_{}.pt'.format(checkpoint_folder, epoch)

        save_checkpoint(generator, generator_optimiser, generator_checkpoint_path, epoch, np.mean(generator_losses))
        save_checkpoint(critic, critic_optimiser, critic_checkpoint_path, epoch, np.mean(critic_losses))
