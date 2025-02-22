import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, noise_dim=100, batch_norm=True):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.fc = nn.Linear(noise_dim, 5 * 5 * 1024)
        self.bn0 = nn.BatchNorm2d(1024) if batch_norm else nn.Identity()
        self.deconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=(3, 3), stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(512) if batch_norm else nn.Identity()
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=(3, 4), stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(256) if batch_norm else nn.Identity()
        self.deconv3 = nn.ConvTranspose2d(256, 1, kernel_size=(4, 6), stride=2, padding=0)
        self.dropout = nn.Dropout(0.5)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 1024, 5, 5)
        x = self.bn0(x)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)
        x = self.dropout(x)
        x = self.deconv1(x)
        x = self.bn1(x)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)
        x = self.dropout(x)
        x = self.deconv2(x)
        x = self.bn2(x)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)
        x = self.dropout(x)
        x = self.deconv3(x)
        x = torch.sigmoid(x)  # Output in [0,1]
        return x

class Discriminator(nn.Module):
    def __init__(self, batch_norm=True):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(4, 5), stride=2, padding=0)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 4), stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(128) if batch_norm else nn.Identity()
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(256) if batch_norm else nn.Identity()
        self.fc = nn.Linear(256 * 5 * 5, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)
        x = self.dropout(x)
        x = x.view(-1, 256 * 5 * 5)
        x = self.fc(x)  # Logits
        return x