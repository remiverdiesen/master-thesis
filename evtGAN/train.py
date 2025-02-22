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

def update_ema(model, ema_model, decay):
    with torch.no_grad():
        for param, ema_param in zip(model.parameters(), ema_model.parameters()):
            ema_param.data = decay * ema_param.data + (1 - decay) * param.data

def train(config):
    cfg = config['model']
    device = torch.device('cuda' if config['training']['use_gpu'] and torch.cuda.is_available() else 'cpu')

    # Load data
    u_train_tensor, _ = load_and_preprocess_data(config)
    u_train_tensor = u_train_tensor.to(device)

    # Initialize models
    generator = Generator(cfg['noise_dim']).to(device)
    discriminator = Discriminator().to(device)
    ema_generator = Generator(cfg['noise_dim']).to(device)
    ema_discriminator = Discriminator().to(device)
    ema_generator.load_state_dict(generator.state_dict())
    ema_discriminator.load_state_dict(discriminator.state_dict())

    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=cfg['learning_rate'], betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=cfg['learning_rate'], betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    # Tracking losses
    g_losses = []
    d_losses = []

    # Training loop
    n_train = u_train_tensor.size(0)
    for epoch in range(cfg['n_epochs']):
        # Train Discriminator 3 times per epoch
        for _ in range(cfg['d_train_it']):
            d_optimizer.zero_grad()
            real_logits = discriminator(u_train_tensor)
            label_real = torch.full((n_train,), 0.9, device=device)  # Label smoothing
            errD_real = criterion(real_logits.squeeze(), label_real)
            noise = torch.randn(n_train, cfg['noise_dim'], device=device)
            fake = generator(noise).detach()
            label_fake = torch.zeros(n_train, device=device)
            fake_logits = discriminator(fake)
            errD_fake = criterion(fake_logits.squeeze(), label_fake)
            errD = errD_real + errD_fake
            errD.backward()
            d_optimizer.step()
            update_ema(discriminator, ema_discriminator, cfg['decay'])

        # Train Generator once per epoch
        g_optimizer.zero_grad()
        noise = torch.randn(n_train, cfg['noise_dim'], device=device)
        fake = generator(noise)
        fake_logits = discriminator(fake)
        errG = criterion(fake_logits.squeeze(), torch.ones(n_train, device=device))
        errG.backward()
        g_optimizer.step()
        update_ema(generator, ema_generator, cfg['decay'])

        # Store losses
        g_losses.append(errG.item())
        d_losses.append(errD.item())

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, D Loss: {errD.item():.4f}, G Loss: {errG.item():.4f}")

    # Save models
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
    plt.plot(range(cfg['n_epochs']), g_losses, label='Generator Loss', color='blue')
    plt.plot(range(cfg['n_epochs']), d_losses, label='Discriminator Loss', color='orange')
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