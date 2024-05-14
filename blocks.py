import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as FV
import torch.optim


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)
        self.bn1 = nn.GroupNorm(out_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(out_channels, out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(8, out_channels)
            )

    def forward(self, x, t):
        t = t.squeeze(-1).squeeze(-1)
        time_emb = self.time_emb_proj(t)
        time_emb = time_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[2], x.shape[3])
        out = self.relu(self.bn1(self.conv1(x)))
        
        out = out +time_emb
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.silu(out)
        return out
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batchsize, C, width, height = x.size()
        proj_query  = self.query_conv(x).view(batchsize, -1, width * height).permute(0, 2, 1)
        proj_key =  self.key_conv(x).view(batchsize, -1, width * height)
        energy =  torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batchsize, C, width, height)
        out = self.gamma*out + x
        return out
    
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super(DownBlock, self).__init__()
        self.residual1 = ResidualBlock(in_channels, out_channels, time_emb_dim)
        self.residual2 = ResidualBlock(out_channels, out_channels, time_emb_dim)
        self.pool = nn.MaxPool2d(2)
        self.attention = SelfAttention(out_channels)

    def forward(self, x, skip_connections, t):
        x = self.residual1(x, t)
        skip_connections.append(x)
        x = self.residual2(x, t)
        x = self.attention(x)
        skip_connections.append(x)
        x = self.pool(x)
        return x
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.residual1 = ResidualBlock(in_channels+out_channels, out_channels, time_emb_dim)
        self.residual2 = ResidualBlock(out_channels*2, out_channels, time_emb_dim)
        self.attention = SelfAttention(out_channels)

    def forward(self, x, skip_connections, t):
        x = self.up(x)
        x= torch.cat((x, skip_connections.pop()), dim=1)
        x = self.residual1(x, t)
        x= torch.cat((x, skip_connections.pop()), dim=1)
        x = self.residual2(x, t)
        x = self.attention(x)
        return x



class UNet(nn.Module):
    def __init__(self, num_classes, device, sinusoidal_embedding, class_emb_size=8, latent_class_emb_size=32, time_emb_dim=32):
        super(UNet, self).__init__()
        self.down1 = DownBlock(64 + latent_class_emb_size, 32, time_emb_dim)
        self.down2 = DownBlock(32, 64,time_emb_dim)
        self.down3 = DownBlock(64, 96,time_emb_dim)
        self.down4 = DownBlock(96, 128,time_emb_dim) 
        self.res1 = ResidualBlock(128, 256, time_emb_dim)
        self.res2 = ResidualBlock(256, 256, time_emb_dim)
        self.attention = SelfAttention(256)
        self.up1 = UpBlock(256, 128, time_emb_dim)
        self.up2 = UpBlock(128, 96, time_emb_dim)
        self.up3 = UpBlock(96, 64, time_emb_dim)
        self.up4 = UpBlock(64, 32, time_emb_dim) 
        self.final_conv = nn.Conv2d(32, 3, kernel_size=1, device=device)
        self.init_conv = nn.Conv2d(3, 32, kernel_size=1, device=device)

        self.class_emb = nn.Embedding(num_classes, class_emb_size)
        self.conditioning_layer = nn.Linear(class_emb_size, latent_class_emb_size)

        self.sinusoidal_embedding = sinusoidal_embedding

    def forward(self, x, noise_variances, class_labels):
        t = self.sinusoidal_embedding(noise_variances)
        noise_embeding = t.expand(-1, -1, 32, 32)
        
        class_embed = self.class_emb(class_labels)
        class_embed = self.conditioning_layer(class_embed)
        class_embed = class_embed.unsqueeze(-1).unsqueeze(-1)
        class_embed = class_embed.expand(-1, -1, x.size(2), x.size(3))

        x = self.init_conv(x)
        x = torch.cat([x, noise_embeding, class_embed], dim=1)

        skip_connections = []
        x = self.down1(x, skip_connections, t)
        x = self.down2(x, skip_connections, t)
        x = self.down3(x, skip_connections, t)
        x = self.down4(x, skip_connections, t)

        x = self.res1(x, t)
        x = self.res2(x, t)
        x = self.attention(x)
        x = self.up1(x, skip_connections, t)
        x = self.up2(x, skip_connections, t)
        x = self.up3(x, skip_connections, t)
        x = self.up4(x, skip_connections, t) 

        x = self.final_conv(x)
        return x
