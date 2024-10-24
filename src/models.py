# src/models.py

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class Generator(nn.Module):
    """
    Generator network for the evtGAN.
    """
    def __init__(self, noise_dim: int, batch_norm: bool = True):
        super(Generator, self).__init__()
        logger.debug(f"Initializing Generator with noise_dim: {noise_dim}.")
        self.noise_dim = noise_dim
        self.batch_norm = batch_norm

        self.fc = nn.Linear(noise_dim, 1024 * 5 * 5, bias=False)
        self.bn0 = nn.BatchNorm2d(1024) if batch_norm else nn.Identity()
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.5)

        self.deconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=(3, 3), stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(512) if batch_norm else nn.Identity()

        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=(3, 4), stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(256) if batch_norm else nn.Identity()

        self.deconv3 = nn.ConvTranspose2d(256, 1, kernel_size=(4, 6), stride=2, padding=0)


    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # 
        x = self.fc(z)
        x = x.reshape(-1, 1024, 5, 5)
        x = self.bn0(x)
        x = self.lrelu(x)
        x = self.dropout(x)
        #
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.lrelu(x)
        x = self.dropout(x)
        #
        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.lrelu(x)
        x = self.dropout(x)
        #
        x = self.deconv3(x)
    
        return x

class Discriminator(nn.Module):
    """
    Discriminator network for the evtGAN.
    """
    def __init__(self, batch_norm: bool = True):
        super(Discriminator, self).__init__()
        logger.debug("Initializing Discriminator.")
        self.batch_norm = batch_norm
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(4, 5), stride=2, padding=0)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 4), stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(128) if batch_norm else nn.Identity()
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(256) if batch_norm else nn.Identity()
        self.fc = nn.Linear(5 * 5 * 256, 1)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.lrelu(x)
        x = self.dropout(x)
        #
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.lrelu(x)
        x = self.dropout(x)
        #
        x = self.conv3(x)
        x = self.bn2(x)
        x = self.lrelu(x)
        x = self.dropout(x)
        # NOTE: Since we are using BCEWithLogitsLoss, which combines a sigmoid layer and binary cross-entropy loss in one, 
        #       we should not apply a sigmoid activation in the Discriminator's output. The Discriminator should output logits directly.
        x = x.reshape(-1, 5 * 5 * 256)
        logits = self.fc(x)
        return logits
