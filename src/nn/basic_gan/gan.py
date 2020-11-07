import torch
from torch import nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime

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


def train(data_loader, noise_dimension, n_epochs,
          device='cpu',
          checkpoint_folder=None,
          critic_repeats=5, lr=0.0001, betas=(0.9, 0.999), gradient_penalty_weight=10):
    if checkpoint_folder is None:
        checkpoint_folder = "models/basic_gan_{}".format(str(datetime.today().date()))
        Path(checkpoint_folder).mkdir(parents=True, exist_ok=True)

    generator_checkpoint_path = '{}/generator.pt'.format(checkpoint_folder)
    critic_checkpoint_path = '{}/critic.pt'.format(checkpoint_folder)

    generator = Generator(noise_dimension).to(device)
    critic = Critic().to(device)

    generator.apply(init_weights)
    critic.apply(init_weights)

    generator_optimiser = torch.optim.Adam(generator.parameters(), lr=lr, betas=betas)
    critic_optimiser = torch.optim.Adam(critic.parameters(), lr=lr, betas=betas)

    generator_losses, critic_losses = [], []

    for epoch in range(n_epochs):
        for real_imgs in tqdm(data_loader):
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

        if epoch <= 10:
            marker = '.'
        else:
            marker = ','

        plt.plot(
            torch.Tensor(generator_losses).view(-1, len(generator_losses)).mean(1),
            label="Generator Loss",
            marker=marker
        )
        plt.plot(
            torch.Tensor(critic_losses).view(-1, len(critic_losses)).mean(1),
            label="Critic Loss",
            marker=marker
        )
        plt.legend()
        plt.show()

        save_checkpoint(generator, generator_optimiser, generator_checkpoint_path, epoch, generator_loss)
        save_checkpoint(critic, critic_optimiser, critic_checkpoint_path, epoch, critic_loss)
