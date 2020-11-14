import matplotlib.pyplot as plt
import numpy as np
import math
import torchvision.utils as vutils
from torch import nn


def plot_batch(real_batch, plot_size=32, caption=None, device='cpu'):
    plt.figure(figsize=(16, 16))
    plt.axis("off")
    plt.imshow(
        np.transpose(
            vutils.make_grid(
                real_batch[:plot_size].detach().cpu().clamp_(0, 1)[:64], padding=2, normalize=True
            ).cpu(), (1, 2, 0)
        )
    )
    if caption is not None:
        plt.title(caption)

    plt.show()


def new_dim_conv2d(h_in, stride, kernel_size, dilation=1, padding=0, out_padding=0):
    return (h_in-1)*stride - 2*padding + dilation*(kernel_size-1)+out_padding+1


def new_dim_conv2d_transpose(h_in, stride, kernel_size, dilation=1, padding=0, out_padding=0):
    return math.floor((h_in + 2*padding +dilation*(kernel_size-1)-1)/stride) + 1


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