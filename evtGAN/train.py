import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from models.evtgan import Generator, Discriminator
from utils.data_utils import load_and_preprocess_data

def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def train(config):
    cfg = config['model']
    device = torch.device('cuda' if config['training']['use_gpu'] and torch.cuda.is_available() else 'cpu')

    # Load data
    u_train_tensor, _ = load_and_preprocess_data(config)
    u_train_tensor = u_train_tensor.to(device)

    # Initialize models
    generator = Generator(cfg['noise_dim']).to(device)
    discriminator = Discriminator().to(device)
    g_optimizer = optim.Adam(generator.parameters(), lr=cfg['learning_rate'], betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=cfg['learning_rate'], betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    n_samples = u_train_tensor.size(0)
    for epoch in range(cfg['n_epochs']):
        # Train Discriminator
        d_optimizer.zero_grad()
        real_logits = discriminator(u_train_tensor)
        z = torch.randn(cfg['batch_size'], 1, 1, cfg['noise_dim'], device=device)
        fake_data = generator(z)
        fake_logits = discriminator(fake_data.detach())
        d_loss_real = criterion(real_logits, 0.9 * torch.ones_like(real_logits))
        d_loss_fake = criterion(fake_logits, torch.zeros_like(fake_logits))
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        g_optimizer.zero_grad()
        z = torch.randn(cfg['batch_size'], 1, 1, cfg['noise_dim'], device=device)
        fake_data = generator(z)
        fake_logits = discriminator(fake_data)
        g_loss = criterion(fake_logits, torch.ones_like(fake_logits))
        g_loss.backward()
        g_optimizer.step()

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    # Save model
    torch.save(generator.state_dict(), config['training']['save_model'])
    print(f"Training complete. Generator saved to {config['training']['save_model']}")

if __name__ == "__main__":
    config = load_config("config/config.yaml")
    train(config)