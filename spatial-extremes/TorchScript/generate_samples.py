#!/usr/bin/env python3
import torch
import torch.nn as nn
from typing import Any

#############################################
# 1. Define the Generator (TorchScript-friendly)
#############################################
class Generator(nn.Module):
    def __init__(self, noise_dim: int, batch_norm: bool = True) -> None:
        super(Generator, self).__init__()
        self.noise_dim: int = noise_dim
        self.batch_norm: bool = batch_norm

        # Fully connected layer to project noise into a 5x5 feature map with 1024 channels
        self.fc = nn.Linear(noise_dim, 1024 * 5 * 5, bias=False)
        self.bn0 = nn.BatchNorm2d(1024) if batch_norm else nn.Identity()
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.5)

        # First deconvolution: from 1024 -> 512 channels
        self.deconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=(3, 3), stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(512) if batch_norm else nn.Identity()

        # Second deconvolution: from 512 -> 256 channels
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=(3, 4), stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(256) if batch_norm else nn.Identity()

        # Final deconvolution: from 256 -> 1 channel (output image)
        self.deconv3 = nn.ConvTranspose2d(256, 1, kernel_size=(4, 6), stride=2, padding=0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z is expected to be of shape [batch_size, noise_dim]
        x = self.fc(z)
        x = x.view(-1, 1024, 5, 5)
        x = self.bn0(x)
        x = self.lrelu(x)
        x = self.dropout(x)

        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.lrelu(x)
        x = self.dropout(x)

        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.lrelu(x)
        x = self.dropout(x)

        x = self.deconv3(x)
        return x

#############################################
# 2. Create a Wrapper Module to Expose Generation
#############################################
class GANGeneratorWrapper(nn.Module):
    def __init__(self, generator: Generator) -> None:
        super(GANGeneratorWrapper, self).__init__()
        self.generator = generator

    @torch.jit.export
    def generate_samples(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Generates synthetic samples from input noise.
        
        Args:
            noise (torch.Tensor): A tensor of shape [batch_size, noise_dim].
        
        Returns:
            torch.Tensor: Generated samples.
        """
        return self.generator(noise)

#############################################
# 3. Main Function: Load Weights, Script, and Save the Model
#############################################
def main() -> None:
    # Set the noise dimension 
    noise_dim = 100  # Adjust if needed
    batch_norm = True

    # Create an instance of the Generator
    gen = Generator(noise_dim=noise_dim, batch_norm=batch_norm)
    try:
        state_dict = torch.load(r"C:\Users\reverd\Repositories\master-thesis\spatial-extremes\experiments\1\model\GEV-GAN\netG_final.pth", map_location="cpu")
        gen.load_state_dict(state_dict)
        print("Pretrained weights loaded successfully.")
    except Exception as e:
        print("Warning: Pretrained weights not loaded. Using randomly initialized weights.", e)

    # Wrap the generator in our wrapper class
    wrapped_model = GANGeneratorWrapper(gen)

    # Script the wrapped model using torch.jit.script
    scripted_model = torch.jit.script(wrapped_model)

    # Save the scripted model to disk; this file is then deployable in SAS Viya.
    torch.jit.save(scripted_model, "scripted_GANGenerator.pt")
    print("Scripted model saved as 'scripted_GANGenerator.pt'.")

    # Test the scripted model with a sample noise input.
    test_noise = torch.randn(4, noise_dim)  # For example, generating 4 samples
    generated_samples = scripted_model.generate_samples(test_noise)
    print("Generated samples shape:", generated_samples.shape)

if __name__ == "__main__":
    main()
