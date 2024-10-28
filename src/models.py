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
        # Add an adaptive pooling layer to ensure a fixed output size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5))

        self.fc = nn.Linear(5 * 5 * 256, 1) # Use pooling to reduce size to (5, 5)

    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Log the initial shape of the input tensor
        # print(f"Input shape: {x.shape}")
        
        # First convolutional layer
        x = self.conv1(x)
        # print(f"After conv1 shape: {x.shape}")
        
        x = self.lrelu(x)
        # print(f"After lrelu (post-conv1) shape: {x.shape}")
        
        x = self.dropout(x)
        # print(f"After dropout (post-conv1) shape: {x.shape}")
        
        # Second convolutional layer
        x = self.conv2(x)
        # print(f"After conv2 shape: {x.shape}")
        
        x = self.bn1(x)
        # print(f"After bn1 (post-conv2) shape: {x.shape}")
        
        x = self.lrelu(x)
        # print(f"After lrelu (post-bn1) shape: {x.shape}")
        
        x = self.dropout(x)
        # print(f"After dropout (post-bn1) shape: {x.shape}")
        
        # Third convolutional layer
        x = self.conv3(x)
        # print(f"After conv3 shape: {x.shape}")
        
        x = self.bn2(x)
        # print(f"After bn2 (post-conv3) shape: {x.shape}")
        
        x = self.lrelu(x)
        # print(f"After lrelu (post-bn2) shape: {x.shape}")
        
        x = self.dropout(x)
        # print(f"After dropout (post-bn2) shape: {x.shape}")
        
        # Use adaptive pooling to get a consistent 5x5 output
        x = self.adaptive_pool(x)
        # Flatten to match the input for the fully connected layer
        x = x.reshape(-1, 5 * 5 * 256)

        # Reshaping before the fully connected layer
        # x = x.reshape(-1, 5 * 5 * 256)
        # x = x.reshape(-1, 256 * 14 * 8)  # Adjust this line

        # print(f"After reshape shape: {x.shape}")
        
        # Fully connected layer
        logits = self.fc(x)
        # print(f"Output logits shape: {logits.shape}\n\n")
        
        return logits

