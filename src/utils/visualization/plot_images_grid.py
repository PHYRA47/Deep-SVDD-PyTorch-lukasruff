import torch
# import matplotlib
# matplotlib.use('Agg')  # or 'PS', 'PDF', 'SVG'

import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid


def plot_images_grid(x: torch.tensor, export_img, title: str = '', nrow=8, padding=2, normalize=False, pad_value=0):
    """Plot 4D Tensor of images of shape (B x C x H x W) as a grid."""

    # Convert hyperspectral data to RGB by selecting specific channels
    if x.shape[1] == 31:  # Check if the tensor has 31 channels (hyperspectral data)

        data = [0.800268, 1.04046, 1.17008] # [0.95047, 1.00000, 1.08883]
        d65_illuminant = torch.tensor(data, device=x.device) 

        red_channel = x[:, 25, :, :].unsqueeze(1) * d65_illuminant[0]
        green_channel = x[:, 15, :, :].unsqueeze(1) * d65_illuminant[1]
        blue_channel = x[:, 5, :, :].unsqueeze(1) * d65_illuminant[2]

        x_rgb = x_rgb / x_rgb.max()

    else:
        x_rgb = x  # For non-hyperspectral data, use the tensor as is

    grid = make_grid(x_rgb, nrow=nrow, padding=padding, normalize=normalize, pad_value=pad_value)
    npgrid = grid.cpu().numpy()

    plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')

    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    if not (title == ''):
        plt.title(title)

    plt.savefig(export_img, bbox_inches='tight', pad_inches=0.1)
    plt.clf()
