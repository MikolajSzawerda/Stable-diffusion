import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import lightning as L
import kornia.augmentation as Kaug
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision import transforms
from PIL import Image

class PreProcess(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transform = transforms.Compose([transforms.ToTensor()])

    @torch.no_grad()
    def forward(self, x: Image) -> torch.Tensor:
        return self.transform(x)
    
class DiffiusionDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "trafic_32", batch_size: int = 256, num_workers: int=10):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.preprocess = PreProcess()

    def setup(self, stage: str):
        dir_dataset = ImageFolder(self.data_dir, transform=self.preprocess)
        self.trainset, self.valset = random_split(dir_dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(42))
        
    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

class ClassSampler:
    def __init__(self, dataset, device):
        targets = torch.tensor([s[1] for s in dataset.samples])
        unique_labels, counts = targets.unique(return_counts=True)
        self.probabilities = counts.float() / counts.sum()
        self.unique_labels = unique_labels
        self.categorical_dist = torch.distributions.Categorical(self.probabilities)
        self.device = device

    def sample(self, x):
        sample_indices = self.categorical_dist.sample([x.size(0)])
        sampled_classes = self.unique_labels[sample_indices]

        return sampled_classes.to(self.device)

def get_sanity_dataset(bs=16, state='train'):
    df = DiffiusionDataModule(batch_size=bs)
    kr = Kaug.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    if state == 'train':
        df.setup('train')
        return df.train_dataloader(), kr
    df.setup('validate')
    return df.val_dataloader(), kr