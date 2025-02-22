import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import os
from datetime import datetime
from models.evtgan import Generator, Discriminator
from utils.data_utils import load_and_preprocess_data
import matplotlib.pyplot as plt

def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def train(config):

    cfg = config['model']
    device = torch.device('cuda' if config['training']['use_gpu'] and torch.cuda.is_available() else 'cpu')

    batch_size = cfg.get('batch_size', 50)      # default to 50
    n_epochs   = cfg.get('n_epochs', 200)       # default to 200
    lr         = cfg.get('learning_rate', 0.0002)
    noise_dim  = cfg.get('noise_dim', 100)

    # Load data 
    u_train_tensor, _ = load_and_preprocess_data(config)
    u_train_tensor = u_train_tensor.to(device)

    # Define Generator and Discriminator
    generator = Generator(noise_dim).to(device)
    discriminator = Discriminator().to(device)

    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # For tracking losses
    g_losses = []
    d_losses = []

    n_train = u_train_tensor.size(0)  
    assert n_train == batch_size, (
        f"Expected the entire dataset to be a single batch of size {batch_size}, "
        f"but got n_train={n_train}."
    )

    print("Starting training...")
    for epoch in range(n_epochs):

        # We do 3 updates of Discriminator + 1 update of Generator each epoch
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        # iterat in range(3):  => D-step each time
        # If iterat % 2 == 0 and iterat != 0 => G-step. That yields 3 D steps and 1 G step.
        for step in range(3):
            # ----------------------------
            #  Train Discriminator
            # ----------------------------
            d_optimizer.zero_grad()

            # Real data forward pass
            real_data = u_train_tensor  
            batch_size_actual = real_data.size(0)

            # Smooth the real labels to 0.9
            label_real = torch.full((batch_size_actual,), 0.9, device=device)
            logits_real = discriminator(real_data).squeeze()
            d_loss_real = criterion(logits_real, label_real)

            # Fake data forward pass
            noise = torch.randn(batch_size_actual, noise_dim, device=device)
            fake_data = generator(noise).detach()
            label_fake = torch.zeros(batch_size_actual, device=device)
            logits_fake = discriminator(fake_data).squeeze()
            d_loss_fake = criterion(logits_fake, label_fake)

            # Total D loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            epoch_d_loss += d_loss.item()

            # ----------------------------
            #  Train Generator
            # ----------------------------
            if step % 2 == 0 and step != 0:
                g_optimizer.zero_grad()
                noise = torch.randn(batch_size_actual, noise_dim, device=device)
                fake_data = generator(noise)
                # For generator, we want the discriminator to predict "real" => label=1.0
                label_g = torch.ones(batch_size_actual, device=device)
                logits_g = discriminator(fake_data).squeeze()
                g_loss = criterion(logits_g, label_g)
                g_loss.backward()
                g_optimizer.step()
                epoch_g_loss += g_loss.item()

        # After the 3 steps in this epoch:
        # - D was updated 3 times
        # - G was updated exactly once
        # Average the losses
        avg_d_loss = epoch_d_loss / 3.0
        # If G was updated once, just store epoch_g_loss as is, or call it avg_g_loss
        avg_g_loss = epoch_g_loss

        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{n_epochs}, D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}")

    # -------------------------------------------------------------------------
    # Save models and losses
    # -------------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join('saved_models', timestamp)
    os.makedirs(save_dir, exist_ok=True)

    torch.save(generator.state_dict(), os.path.join(save_dir, 'generator_weights.pth'))
    with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    scripted_generator = torch.jit.script(generator)
    scripted_generator.save(os.path.join(save_dir, 'generator.pt'))

    # Plot and save losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(n_epochs), g_losses, label='Generator Loss', color='blue')
    plt.plot(range(n_epochs), d_losses, label='Discriminator Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator and Discriminator Losses Over Training')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(save_dir, 'losses_plot.png')
    plt.savefig(plot_path)
    plt.close()

    print(f"Training complete. Models and loss plot saved in {save_dir}")

if __name__ == "__main__":
    config = load_config("config/config.yaml")
    train(config)
