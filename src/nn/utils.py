import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils


def plot_batch(real_batch, plot_size=32, caption=None, device='cpu'):
    plt.figure(figsize=(16, 16))
    plt.axis("off")
    plt.imshow(
        np.transpose(
            vutils.make_grid(
                real_batch[:plot_size].to(device)[:64], padding=2, normalize=True
            ).cpu(), (1, 2, 0)
        )
    )
    if caption is not None:
        plt.title(caption)

    plt.show()
