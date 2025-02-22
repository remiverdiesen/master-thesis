import argparse
import os
import yaml
import torch
import numpy as np
import pandas as pd
from models.evtgan import Generator
from utils.data_utils import load_netcdf
from utils.evt_utils import fit_gev, transform_back

def load_config(model_dir):
    config_path = os.path.join('saved_models', model_dir, 'config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def generate_samples(model_dir, model_type, n_gen, device):
    config = load_config(model_dir)
    cfg = config['model']

    # Load training data for GEV fitting, discarding NaN values
    train_data_full = load_netcdf(config['data']['train_file'])  # [n_samples, 18, 22]
    # Remove rows with any NaN values
    mask = ~np.any(np.isnan(train_data_full), axis=(1, 2))  # True where no NaN in row
    train_data_clean = train_data_full[mask]
    # Limit to n_train samples as in training
    n_train = cfg['n_train']
    if train_data_clean.shape[0] < n_train:
        raise ValueError(f"After removing NaN values, only {train_data_clean.shape[0]} samples remain, but n_train={n_train} is required.")
    train_data = train_data_clean[:n_train]  # Ensures we use the same subset as training
    print(f"Using {train_data.shape[0]} clean samples for GEV fitting.")
    
    # Fit GEV parameters
    params = fit_gev(train_data)

    # Load generator
    if model_type == 'pickle':
        generator = Generator(cfg['noise_dim']).to(device)
        weights_path = os.path.join('saved_models', model_dir, 'generator_weights.pth')
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found at {weights_path}")
        generator.load_state_dict(torch.load(weights_path, map_location=device))
    elif model_type == 'torchscript':
        scripted_path = os.path.join('saved_models', model_dir, 'generator.pt')
        if not os.path.exists(scripted_path):
            raise FileNotFoundError(f"TorchScript model not found at {scripted_path}")
        generator = torch.jit.load(scripted_path, map_location=device)
    else:
        raise ValueError("model_type must be 'pickle' or 'torchscript'")
    generator.eval()

    # Generate samples
    with torch.no_grad():
        z = torch.randn(n_gen, cfg['noise_dim'], device=device)
        u_gen_padded = generator(z).cpu().numpy()  # [n_gen, 1, 20, 24]
        u_gen = u_gen_padded[:, :, 1:19, 1:23]    # [n_gen, 1, 18, 22]

    # Transform back to original scale
    z_gen = transform_back(u_gen, params)  # [n_gen, 18, 22]
    z_gen_flat = z_gen.reshape(n_gen, -1)  # [n_gen, 396]

    # Save to CSV
    output_dir = config['data']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"synthetic_precipitation_{model_dir}.csv")
    pd.DataFrame(z_gen_flat).to_csv(output_file, header=None, index=False)
    print(f"Generated {n_gen} samples saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic precipitation samples using evtGAN.")
    parser.add_argument('--model_dir', type=str, required=True, help="Subdirectory under saved_models/")
    parser.add_argument('--model_type', type=str, choices=['pickle', 'torchscript'], required=True, 
                        help="Type of model to load: 'pickle' or 'torchscript'")
    parser.add_argument('--n_gen', type=int, default=10000, help="Number of samples to generate")
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generate_samples(args.model_dir, args.model_type, args.n_gen, device)