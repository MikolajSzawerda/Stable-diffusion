import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as FV
import torch.optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from tqdm import tqdm
import os

import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import lightning as L
from torchmetrics import Accuracy
from lightning.pytorch.loggers import TensorBoardLogger
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import kornia.augmentation as Kaug
import kornia as K
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from torchvision import transforms
from PIL import Image
import os
from torchmetrics.image.fid import FrechetInceptionDistance
import optuna

def sinusoidal_embedding(x, device):
    log_min_freq = math.log(1.0)
    log_max_freq = math.log(1000.0)
    num_frequencies = 16
    frequencies = torch.exp(torch.linspace(log_min_freq, log_max_freq, num_frequencies, dtype=torch.float32)).to(device)
    angular_speeds = 2.0 * math.pi * frequencies
    sin_embeddings = torch.sin(angular_speeds.unsqueeze(1) * x).to(device)
    cos_embeddings = torch.cos(angular_speeds.unsqueeze(1) * x).to(device)
    embeddings = torch.cat([sin_embeddings, cos_embeddings], dim=1)
    return embeddings.view(-1, 32, 1, 1)

def cosine_diffusion_schedule(diffusion_times):
    min_signal_rate = torch.tensor(0.00, device=DEVICE)
    max_signal_rate = torch.tensor(0.99, device=DEVICE)
    start_angle = torch.acos(max_signal_rate).to(DEVICE)
    end_angle = torch.acos(min_signal_rate).to(DEVICE)
    angles = start_angle+diffusion_times*(end_angle-start_angle)
    signal_rates = torch.cos(angles).to(DEVICE)
    noise_rates = torch.sin(angles).to(DEVICE)
    return noise_rates, signal_rates

class LinearNoiseScheduler:
    def __init__(self, num_steps, device, beta_start=1e-4, beta_end=2e-2):
        
        self.device = device
        
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_steps, dtype=torch.float32, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
    def get_alpha_bar(self, t):
        return self.alpha_bars[t]
    
    def get_variance(self, t):
        return self.betas[t]
    
    def get_rates(self, t):
        alpha_bar_t = self.get_alpha_bar(t)
        return (
            torch.sqrt(alpha_bar_t).to(self.device).unsqueeze(1).unsqueeze(2).unsqueeze(3),
            torch.sqrt(1 - alpha_bar_t).to(self.device).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        )
    
    def add_noise(self, x, noise, t):
        alpha_bar_t = self.get_alpha_bar(t)
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return (sqrt_alpha_bar_t * x + sqrt_one_minus_alpha_bar_t * noise), sqrt_alpha_bar_t, sqrt_one_minus_alpha_bar_t

