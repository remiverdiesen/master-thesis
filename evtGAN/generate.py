import torch
import numpy as np
import pandas as pd
import yaml
import os
from models.evtgan import Generator
from utils.data_utils import load_netcdf
from utils.evt_utils import fit_gev, transform_back

def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def generate_samples(config, n_gen=10000):
    cfg = config['model']
    device = torch.device('cuda' if config['training']['use_gpu'] and torch.cuda.is_available() else 'cpu')

    # Load training data for GEV fitting
    train_data = load_netcdf(config['data']['train_file'])
    params = fit_gev(train_data)

    # Load trained generator
    generator = Generator(cfg['noise_dim']).to(device)
    generator.load_state_dict(torch.load(config['training']['save_model'], map_location=device))
    generator.eval()

    # Generate samples
    with torch.no_grad():
        z = torch.randn(n_gen, 1, 1, cfg['noise_dim'], device=device)
        u_gen_padded = generator(z).cpu().numpy()  # [n_gen, 1, 20, 24]
        u_gen = u_gen_padded[:, :, 1:19, 1:23]    # [n_gen, 1, 18, 22]

    # Transform back to original scale
    z_gen = transform_back(u_gen, params)  # [n_gen, 18, 22]
    z_gen_flat = z_gen.reshape(n_gen, -1)  # [n_gen, 396]

    # Save to CSV
    os.makedirs(config['data']['output_dir'], exist_ok=True)
    output_file = f"{config['data']['output_dir']}/synthetic_precipitation.csv"
    pd.DataFrame(z_gen_flat).to_csv(output_file, header=None, index=False)
    print(f"Generated {n_gen} samples saved to {output_file}")

if __name__ == "__main__":
    config = load_config("config/config.yaml")
    generate_samples(config)