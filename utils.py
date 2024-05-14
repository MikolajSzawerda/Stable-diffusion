import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.transforms.functional as FV
import torch.optim

from torchvision.utils import make_grid



def show(imgs, denorm = True):
    if denorm:
        mean = torch.tensor((0.5, 0.5, 0.5)).view(1, 3, 1, 1)
        std = torch.tensor((0.5, 0.5, 0.5)).view(1, 3, 1, 1)
        imgs = imgs*std+mean
    imgs = make_grid(imgs)
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = FV.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])