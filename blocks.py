import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as FV
import torch.optim


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(8, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(8, out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(8, out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out
    
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.residual1 = ResidualBlock(in_channels, out_channels)
        self.residual2 = ResidualBlock(out_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, skip_connections):
        x = self.residual1(x)
        skip_connections.append(x)
        x = self.residual2(x)
        skip_connections.append(x)
        x = self.pool(x)
        return x
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.residual1 = ResidualBlock(in_channels+out_channels, out_channels)
        self.residual2 = ResidualBlock(out_channels*2, out_channels)

    def forward(self, x, skip_connections):
        x = self.up(x)
        x= torch.cat((x, skip_connections.pop()), dim=1)
        x = self.residual1(x)
        x= torch.cat((x, skip_connections.pop()), dim=1)
        x = self.residual2(x)
        return x

class UNet(nn.Module):
    def __init__(self, num_classes, device, sinusoidal_embedding, class_emb_size=8, latent_class_emb_size=32):
        super(UNet, self).__init__()
        self.down1 = DownBlock(64+latent_class_emb_size, 32)
        self.down2 = DownBlock(32, 64)
        self.down3 = DownBlock(64, 96)
        self.res1 = ResidualBlock(96, 128)
        self.res2 = ResidualBlock(128, 128)
        self.up1 = UpBlock(128, 96)
        self.up2 = UpBlock(96, 64)
        self.up3 = UpBlock(64, 32)
        self.final_conv = nn.Conv2d(32, 3, kernel_size=1, device=device)
        self.init_conv = nn.Conv2d(3, 32, kernel_size=1, device=device)
        
        self.class_emb = nn.Embedding(num_classes, class_emb_size)
        self.conditioning_layer = nn.Linear(class_emb_size, latent_class_emb_size)
        
        self.sinusoidal_embedding = sinusoidal_embedding

    def forward(self, x, noise_variances, class_labels):
        noise_embeding = self.sinusoidal_embedding(noise_variances)
        noise_embeding = noise_embeding.expand(-1, -1, 32, 32)
        
        class_embed = self.class_emb(class_labels)
        class_embed = self.conditioning_layer(class_embed)
        class_embed = class_embed.unsqueeze(-1).unsqueeze(-1)
        class_embed = class_embed.expand(-1, -1, x.size(2), x.size(3))

        x = self.init_conv(x)
        x = torch.cat([x, noise_embeding, class_embed], dim=1)

        skip_connections = []
        x = self.down1(x, skip_connections)
        x = self.down2(x, skip_connections)
        x = self.down3(x, skip_connections)
        
        x = self.res1(x)
        x = self.res2(x)
        x = self.up1(x, skip_connections)
        x = self.up2(x, skip_connections)
        x = self.up3(x, skip_connections)

        x = self.final_conv(x)
        return x
