import matplotlib.pyplot as plt
import numpy as np
import math
import torchvision.utils as vutils


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


def new_dim_conv2d(h_in, stride, kernel_size):
    dilation=1
    padding = 0
    out_padding=0
    return (h_in-1)*stride - 2*padding + dilation*(kernel_size-1)+out_padding+1


def new_dim_conv2d_transpose(h_in, stride, kernel_size):
    dilation=1
    padding = 0
    out_padding=0
    return math.floor((h_in + 2*padding +dilation*(kernel_size-1)-1)/stride) + 1
