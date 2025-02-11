# src/train.py

import os
import time
import logging
import torch
import torch.optim as optim
import numpy as np
import copy
import matplotlib.pyplot as plt  # Add this import for plotting

from config import Config
from data_handler import DataHandler
from models import Generator, Discriminator
from utils import weights_init, update_ema, compute_EC_loss, plot_loss_curve

# =============================================================================
# Set up logging
# =============================================================================
logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s - line: %(lineno)d - %(message)s "
)
logger = logging.getLogger(__name__)

# =============================================================================
# Training Function
# =============================================================================

def train():
    logger.info("\n\n\n Setting up the config...\n")
    config = Config()
    Config.save_config(config)

    # Initialize DataHandler
    datasets = DataHandler(config)

    # Initialize Networks
    netG = Generator(config.noise_dim, batch_norm=config.batch_norm).to(config.device)
    netD = Discriminator(batch_norm=config.batch_norm).to(config.device)

    netG.apply(weights_init)
    netD.apply(weights_init)

    # Set up optimizers
    optimizerD = optim.Adam(netD.parameters(), lr=config.learning_rate, betas=(config.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=config.learning_rate, betas=(config.beta1, 0.999))

    # Loss function
    criterion = torch.nn.BCEWithLogitsLoss()

    # Exponential Moving Average (EMA) models
    ema_decay = config.decay
    ema_G = copy.deepcopy(netG)
    ema_D = copy.deepcopy(netD)

    # Prepare data
    train_set = datasets.U_train
    n_train = train_set.shape[0]
    batch_size = config.batch_size

    # Labels
    real_label = 1.0
    fake_label = 0.0
    smooth_real_label = 0.9  # Apply label smoothing

    # Initialize lists to store loss values per epoch
    generator_losses = []
    discriminator_losses = []
    logger.info(f"\n\n\n Training the {config.model_type}-GAN...\n")
    # Training loop
    num_batches = n_train // batch_size
    for epoch in range(config.train_epoch):
        epoch_start_time = time.time()
        epoch_loss_G = 0.0
        epoch_loss_D = 0.0
        for i in range(num_batches):
            # ---------------------------
            # Update Discriminator
            # ---------------------------
            netD.zero_grad()

            # Train with real data
            data = train_set[i * batch_size:(i + 1) * batch_size]
            batch_size_actual = data.size(0)
            data = data.to(config.device)

            # Create labels for real data with label smoothing
            label_real = torch.full((batch_size_actual,), smooth_real_label, device=config.device)

            # Forward pass real batch through D
            logits_real = netD(data)
            errD_real = criterion(logits_real.squeeze(), label_real)

            # Train with fake data
            noise = torch.randn(batch_size_actual, config.noise_dim, device=config.device)
            fake = netG(noise).detach()  # Detach to avoid backprop into G
            label_fake = torch.full((batch_size_actual,), fake_label, device=config.device)

            logits_fake = netD(fake)
            errD_fake = criterion(logits_fake.squeeze(), label_fake)

            # Combine losses and perform backward pass
            errD = errD_real + errD_fake
            errD.backward()
            optimizerD.step()

            epoch_loss_D += errD.item()
            # ---------------------------
            # Update Generator
            # ---------------------------
            if i % config.D_train_it == 0:
                netG.zero_grad()

                # Generate new fake data
                noise = torch.randn(batch_size_actual, config.noise_dim, device=config.device)
                fake = netG(noise)
                logits_fake_g = netD(fake)

                # Non-saturating heuristic loss for generator
                errG_adv = -torch.mean(torch.log(torch.sigmoid(logits_fake_g) + 1e-8))

                if config.use_EC_regul: # 
                    # EC regularization
                    real_data_for_EC = data.clone().detach()
                    EC_loss = compute_EC_loss(fake, real_data_for_EC, datasets.ids_, datasets.pos_coordinates, config)
                    errG = errG_adv + config.LAMBDA * EC_loss
                else:
                    errG = errG_adv  # Do not add LAMBDA when EC regularization is disabled

                # Backward pass and optimization
                errG.backward()
                optimizerG.step()

                epoch_loss_G += errG.item()

                # Update EMA
                update_ema(netG, ema_G, ema_decay)
                update_ema(netD, ema_D, ema_decay)
        
        # Average losses over the epoch
        avg_loss_G = epoch_loss_G / num_batches
        avg_loss_D = epoch_loss_D / num_batches

        # Store the epoch losses for plotting
        generator_losses.append(avg_loss_G)
        discriminator_losses.append(avg_loss_D)

        # Logging
        epoch_end_time = time.time()
        per_epoch_time = epoch_end_time - epoch_start_time
        logger.info(f"Epoch [{epoch + 1}/{config.train_epoch}] - errG: {errG.item():.4f} - errD: {errD.item():.4f} - Time: {per_epoch_time:.2f}s")

        # Save models at certain intervals
        if (epoch + 1) % 50 == 0 or epoch == config.train_epoch - 1:
            model_dir = config.models_dir
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            torch.save(netG.state_dict(), os.path.join(model_dir,f'{config.model_type}-GAN', f'netG_epoch_{epoch+1}.pth'))
            torch.save(netD.state_dict(), os.path.join(model_dir,f'{config.model_type}-GAN', f'netD_epoch_{epoch+1}.pth'))
            logger.info(f"Saved models at epoch {epoch+1}")

    # Plot the loss curve
    plot_loss_curve(generator_losses, discriminator_losses, config)

    # Save final models
    torch.save(netG.state_dict(), os.path.join(config.models_dir, 'netG_final.pth'))
    torch.save(netD.state_dict(), os.path.join(config.models_dir, 'netD_final.pth'))
    logger.info("Training complete and final models saved.")



if __name__ == "__main__":
    train()
