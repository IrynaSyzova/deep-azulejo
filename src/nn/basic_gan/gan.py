import torch
from torch import nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

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
        nn.init.constant(layer.bias, 0)


def train(dataloader, noise_dimention, n_epochs,
          device='cpu',
          critic_repeats=5, lr=0.0001, betas=(0.9, 0.999), gradient_penalty_weight=10,
          display_step=10000):
    generator = Generator(noise_dimention).to(device)
    critic = Critic().to(device)

    generator.apply(init_weights)
    critic.apply(init_weights)

    generator_optimiser = torch.optim.Adam(generator.parameters(), lr=lr, betas=betas)
    critic_optimiser = torch.optim.Adam(critic.parameters(), lr=lr, betas=betas)

    generator_losses, critic_losses = [], []

    step = 0

    for epoch in range(n_epochs):
        for real_imgs in tqdm(dataloader):
            batch_size = len(real_imgs)
            real_imgs = real_imgs.to(device)

            mean_critic_loss = 0
            for _ in range(critic_repeats):
                critic_optimiser.zero_grad()

                fake_imgs = generator(generator.get_noise(batch_size, noise_dimention, device=device))

                critic_loss = critic.loss(fake_imgs, real_imgs, gradient_penalty_weight, device=device)

                mean_critic_loss += critic_loss.item() / critic_repeats

                critic_loss.backward(retain_graph=True)
                # Update optimizer
                critic_optimiser.step()

            critic_losses.append(mean_critic_loss)

            generator_optimiser.zero_grad()

            fake_imgs = generator(generator.get_noise(batch_size, noise_dimention, device=device))
            fake_pred = critic(fake_imgs)

            generator_loss = generator.loss(fake_pred)
            generator_loss.backward()
            generator_optimiser.step()
            generator_losses.append(generator_loss.item())

            if step % display_step == 0:
                # Visualisation

                print("Step {step}: Generator loss: {gen_mean}, critic loss: {crit_mean}".format(
                    step=step,
                    gen_mean=sum(generator_losses[-display_step:]) / display_step,
                    crit_mean=sum(critic_losses[-display_step:]) / display_step
                ))
                plot_batch(fake_imgs, device=device, caption='Generated images')
                plot_batch(real_imgs, device=device, caption='Real images')

                step_bins = 100
                num_examples = (len(generator_losses) // step_bins) * step_bins
                plt.plot(
                    range(num_examples // step_bins),
                    torch.Tensor(generator_losses[:num_examples]).view(-1, step_bins).mean(1),
                    label="Generator Loss"
                )
                plt.plot(
                    range(num_examples // step_bins),
                    torch.Tensor(critic_losses[:num_examples]).view(-1, step_bins).mean(1),
                    label="Critic Loss"
                )
                plt.legend()
                plt.show()

        step += 1
