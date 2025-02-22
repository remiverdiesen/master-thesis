import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import os
from models.evtgan import Generator, Discriminator
from utils.data_utils import load_and_preprocess_data, get_relevant_points
from utils.evt_utils import get_pos_coordinates, compute_ECs

def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def train(config):
    cfg = config['model']
    device = torch.device('cuda' if config['training']['use_gpu'] and torch.cuda.is_available() else 'cpu')

    # Load data
    train_data, ids = load_and_preprocess_data(config['data']['data_file'], config['data']['ids_file'])
    train_data = train_data.to(device)
    ids = ids.to(device)

    # Initialize models
    generator = Generator(cfg['noise_dim']).to(device)
    discriminator = Discriminator().to(device)
    g_ema = Generator(cfg['noise_dim']).to(device)
    g_ema.load_state_dict(generator.state_dict())
    d_ema = Discriminator().to(device)
    d_ema.load_state_dict(discriminator.state_dict())

    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=cfg['learning_rate'], betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=cfg['learning_rate'], betas=(0.5, 0.999))

    # Loss functions
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    pos_coords = get_pos_coordinates(cfg['n_sub_ids']).to(device)
    num_pairs = len(pos_coords)
    for epoch in range(cfg['n_epochs']):
        z = torch.randn(cfg['batch_size'], cfg['noise_dim']).to(device)
        real_batch = train_data[:cfg['batch_size']]

        # Train Discriminator
        d_optimizer.zero_grad()
        real_logits = discriminator(real_batch)
        fake_data = generator(z)
        fake_logits = discriminator(fake_data.detach())
        d_loss = criterion(real_logits, torch.ones_like(real_logits) * 0.9) + criterion(fake_logits, torch.zeros_like(fake_logits))
        d_loss.backward()
        d_optimizer.step()

        # Update D EMA
        for param, ema_param in zip(discriminator.parameters(), d_ema.parameters()):
            ema_param.data = cfg['decay'] * ema_param.data + (1 - cfg['decay']) * param.data
        discriminator.load_state_dict(d_ema.state_dict())

        if epoch % cfg['d_train_it'] == 0:
            # Train Generator
            g_optimizer.zero_grad()
            fake_data = generator(z)
            fake_logits = discriminator(fake_data)
            g_loss_no_reg = criterion(fake_logits, torch.ones_like(fake_logits))

            # EVT regularization
            ind_loss = torch.tensor(np.random.choice(ids.size(0), cfg['n_sub_ids'], replace=False), dtype=torch.long).to(device)
            real_points = get_relevant_points(real_batch, ids)[:, ind_loss]
            fake_points = get_relevant_points(fake_data, ids)[:, ind_loss]
            gEC = compute_ECs(fake_points, pos_coords)
            trEC = compute_ECs(real_points, pos_coords)
            reg_loss = cfg['lambda_reg'] * (1 / num_pairs ** 0.5) * torch.norm(gEC - trEC)
            g_loss = g_loss_no_reg + reg_loss
            g_loss.backward()
            g_optimizer.step()

            # Update G EMA
            for param, ema_param in zip(generator.parameters(), g_ema.parameters()):
                ema_param.data = cfg['decay'] * ema_param.data + (1 - cfg['decay']) * param.data
            generator.load_state_dict(g_ema.state_dict())

        # Save checkpoints
        save_points = [cfg['n_epochs'] // 4 - 1, cfg['n_epochs'] // 2 - 1, cfg['n_epochs'] * 3 // 4 - 1, cfg['n_epochs'] - 1]
        if epoch in save_points:
            torch.save(generator.state_dict(), f"data/synthetic/generator_epoch_{epoch}.pt")
            print(f"Saved model at epoch {epoch}")

if __name__ == "__main__":
    config = load_config("config/config.yaml")
    train(config)