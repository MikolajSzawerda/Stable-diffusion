import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as FV
import torch.optim


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, class_emb_dim, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels+out_channels*2, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)
        self.cls_emb_proj = nn.Linear(class_emb_dim, out_channels)
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

    def forward(self, x, t, c):
        t = t.squeeze(-1).squeeze(-1)
        c = c.squeeze(-1).squeeze(-1)
        
        time_emb = F.relu(self.time_emb_proj(t))
        time_emb = time_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[2], x.shape[3])

        cls_emb = F.relu(self.cls_emb_proj(c))
        # print(time_emb.shape, cls_emb.shape)
        cls_emb = cls_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[2], x.shape[3])
        out = torch.cat([x, time_emb, cls_emb], dim=1)
        out = self.relu(self.bn1(self.conv1(out)))
        
        # out = out +time_emb+cls_emb
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
    def __init__(self, in_channels, out_channels, time_emb_dim, class_emb_dim):
        super(DownBlock, self).__init__()
        self.residual1 = ResidualBlock(in_channels, out_channels, time_emb_dim, class_emb_dim)
        self.residual2 = ResidualBlock(out_channels, out_channels, time_emb_dim, class_emb_dim)
        self.pool = nn.MaxPool2d(2)
        self.attention = SelfAttention(out_channels)

    def forward(self, x, skip_connections, t, c):
        x = self.residual1(x, t, c)
        skip_connections.append(x)
        x = self.residual2(x, t, c)
        x = self.attention(x)
        skip_connections.append(x)
        x = self.pool(x)
        return x
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, class_emb_dim):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.residual1 = ResidualBlock(in_channels+out_channels, out_channels, time_emb_dim, class_emb_dim)
        self.residual2 = ResidualBlock(out_channels*2, out_channels, time_emb_dim, class_emb_dim)
        self.attention = SelfAttention(out_channels)

    def forward(self, x, skip_connections, t, c):
        x = self.up(x)
        x= torch.cat((x, skip_connections.pop()), dim=1)
        x = self.residual1(x, t, c)
        x= torch.cat((x, skip_connections.pop()), dim=1)
        x = self.residual2(x, t, c)
        x = self.attention(x)
        return x



class UNet(nn.Module):
    def __init__(self, num_classes, device, sinusoidal_embedding, class_emb_size=8, latent_class_emb_size=32, time_emb_dim=32):
        super(UNet, self).__init__()
        self.down1 = DownBlock(64, 64, time_emb_dim, class_emb_size)
        self.down2 = DownBlock(64, 128,time_emb_dim, class_emb_size)
        self.down3 = DownBlock(128, 256,time_emb_dim, class_emb_size)
        self.down4 = DownBlock(256, 512,time_emb_dim, class_emb_size) 
        self.res1 = ResidualBlock(512, 1024, time_emb_dim, class_emb_size)
        self.res2 = ResidualBlock(1024, 1024, time_emb_dim, class_emb_size)
        self.attention = SelfAttention(1024)
        self.up1 = UpBlock(1024, 512, time_emb_dim, class_emb_size)
        self.up2 = UpBlock(512, 256, time_emb_dim, class_emb_size)
        self.up3 = UpBlock(256, 128, time_emb_dim, class_emb_size)
        self.up4 = UpBlock(128, 64, time_emb_dim, class_emb_size) 
        self.final_conv = nn.Conv2d(64, 3, kernel_size=1, device=device)
        self.init_conv = nn.Conv2d(3, 32, kernel_size=1, device=device)
        self.device = device
        self.class_emb = nn.Embedding(num_classes, class_emb_size)
        self.sinusoidal_embedding = sinusoidal_embedding

    def forward(self, x, noise_variances, class_labels):
        t = self.sinusoidal_embedding(noise_variances, self.device)
        noise_embeding = t.expand(-1, -1, 32, 32)
        c = self.class_emb(class_labels)

        x = self.init_conv(x)
        x = torch.cat([x, noise_embeding], dim=1)

        skip_connections = []
        x = self.down1(x, skip_connections, t, c)
        x = self.down2(x, skip_connections, t, c)
        x = self.down3(x, skip_connections, t, c)
        x = self.down4(x, skip_connections, t, c)

        x = self.res1(x, t, c)
        x = self.res2(x, t, c)
        x = self.attention(x)
        x = self.up1(x, skip_connections, t, c)
        x = self.up2(x, skip_connections, t, c)
        x = self.up3(x, skip_connections, t, c)
        x = self.up4(x, skip_connections, t, c) 

        x = self.final_conv(x)
        return x
