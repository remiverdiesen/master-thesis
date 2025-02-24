import argparse
import os
import yaml
import torch
import numpy as np
import pandas as pd
from scipy.stats import genextreme, expon
from datetime import datetime, timedelta
from models.evtgan import Generator
from utils.data_utils import load_netcdf
from utils.evt_utils import fit_gev, transform_back

def load_config(model_dir):
    config_path = os.path.join('saved_models', model_dir, 'config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def generate_samples(model_dir, model_type, n_gen=None, years=None, device='cpu'):
    # Validate input: either n_gen or years must be provided, not both
    if (n_gen is None and years is None) or (n_gen is not None and years is not None):
        raise ValueError("Specify either --n_gen for annual maxima or --years for daily rainfall simulation, but not both.")
    if years is not None and years < 5:
        raise ValueError("Number of years must be at least 5 for daily rainfall simulation.")

    config = load_config(model_dir)
    cfg = config['model']

    # Load training data for GEV fitting, discarding NaN values
    train_data_full = load_netcdf(config['data']['train_file'])  # [n_samples, 18, 22]
    mask = ~np.any(np.isnan(train_data_full), axis=(1, 2))  # True where no NaN in row
    train_data_clean = train_data_full[mask]
    n_train = cfg['n_train']
    if train_data_clean.shape[0] < n_train:
        raise ValueError(f"After removing NaN values, only {train_data_clean.shape[0]} samples remain, but n_train={n_train} is required.")
    train_data = train_data_clean[:n_train]
    print(f"Using {train_data.shape[0]} clean samples for GEV fitting.")

    # Fit GEV parameters
    params = fit_gev(train_data)  # [18, 22, 3]

    # Load IDs file
    ids_file = "data/ids.csv"
    if not os.path.exists(ids_file):
        raise FileNotFoundError(f"IDs file not found at {ids_file}")
    ids_df = pd.read_csv(ids_file)
    assert len(ids_df) == 396, f"Expected 396 grid points in {ids_file}, got {len(ids_df)}"

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

    # Determine generation mode
    if n_gen is not None:
        # Mode 1: Generate annual maxima
        with torch.no_grad():
            z = torch.randn(n_gen, cfg['noise_dim'], device=device)
            u_gen_padded = generator(z).cpu().numpy()  # [n_gen, 1, 20, 24]
            u_gen = u_gen_padded[:, :, 1:19, 1:23]    # [n_gen, 1, 18, 22]
        
        # Transform back to original scale
        z_gen = transform_back(u_gen, params)  # [n_gen, 18, 22]
        z_gen_flat = z_gen.reshape(n_gen, -1)  # [n_gen, 396]

        # Prepare output DataFrame for annual maxima
        dates = [datetime(2026, 1, 1) + timedelta(days=i) for i in range(n_gen)]
        output_data = []
        for sample_idx in range(n_gen):
            for _, row in ids_df.iterrows():
                region_id = int(row['Region_ID'])  # Convert to integer
                output_data.append([
                    int(row['I']),
                    int(row['J']),
                    region_id,
                    float(row['Latitude']),
                    float(row['Longitude']),
                    dates[sample_idx].strftime('%Y-%m-%d'),
                    float(z_gen_flat[sample_idx, region_id - 1])  # Use integer index
                ])
        output_df = pd.DataFrame(output_data, columns=['I', 'J', 'Region_ID', 'Latitude', 'Longitude', 'Date', 'Value'])
        output_suffix = f"annual_maxima_{n_gen}samples_{model_dir}"
        description = f"Generated {n_gen} annual maxima samples"

    elif years is not None:
        # Mode 2: Simulate daily rainfall
        n_days_per_year = 365
        n_days = years * n_days_per_year

        # Generate annual maxima
        with torch.no_grad():
            z = torch.randn(years, cfg['noise_dim'], device=device)
            u_gen_padded = generator(z).cpu().numpy()  # [years, 1, 20, 24]
            u_gen = u_gen_padded[:, :, 1:19, 1:23]    # [years, 1, 18, 22]
            annual_maxima = np.zeros((years, 18, 22))
            for i in range(18):
                for j in range(22):
                    c, loc, scale = params[i, j]
                    u = u_gen[:, 0, i, j]
                    annual_maxima[:, i, j] = genextreme.ppf(u, c, loc=loc, scale=scale)

        # Simulate daily rainfall
        daily_rainfall = np.zeros((n_days, 18, 22))
        for year in range(years):
            year_start = year * n_days_per_year
            year_end = (year + 1) * n_days_per_year
            year_maxima = annual_maxima[year]

            for i in range(18):
                for j in range(22):
                    max_value = year_maxima[i, j]
                    if max_value <= 0:
                        daily_rainfall[year_start:year_end, i, j] = 0
                        continue
                    mean_daily = max_value / 50
                    scale = mean_daily
                    daily_values = expon.rvs(scale=scale, size=n_days_per_year - 1)
                    daily_values = np.clip(daily_values, 0, max_value * 0.99)
                    max_day = np.random.randint(0, n_days_per_year)
                    daily_values = np.concatenate([daily_values[:max_day], [max_value], daily_values[max_day:]])
                    daily_rainfall[year_start:year_end, i, j] = daily_values[:n_days_per_year]

        # Prepare output DataFrame for daily rainfall
        start_date = datetime(2026, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(n_days)]
        output_data = []
        for day_idx in range(n_days):
            for _, row in ids_df.iterrows():
                i_idx = int(row['I']) - 1  # column index
                j_idx = int(row['J']) - 1  # row index
                region_id = int(row['Region_ID'])
                value = float(daily_rainfall[day_idx, j_idx, i_idx])  # Swapped indices
                output_data.append([
                    int(row['I']),
                    int(row['J']),
                    region_id,
                    float(row['Latitude']),
                    float(row['Longitude']),
                    dates[day_idx].strftime('%Y-%m-%d'),
                    value
                ])
        output_df = pd.DataFrame(output_data, columns=['I', 'J', 'Region_ID', 'Latitude', 'Longitude', 'Date', 'Value'])
        output_suffix = f"daily_rainfall_{years}years_{model_dir}"
        description = f"Simulated {n_days} daily rainfall samples over {years} years"

    # Save to CSV
    output_dir = config['data']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{output_suffix}.csv")
    output_df.to_csv(output_file, index=False)
    print(f"{description} saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic precipitation samples or daily rainfall using evtGAN.")
    parser.add_argument('--model_dir', type=str, required=True, help="Subdirectory under saved_models/")
    parser.add_argument('--model_type', type=str, choices=['pickle', 'torchscript'], required=True, 
                        help="Type of model to load: 'pickle' or 'torchscript'")
    parser.add_argument('--n_gen', type=int, default=None, help="Number of annual maxima samples to generate")
    parser.add_argument('--years', type=int, default=None, help="Number of years for daily rainfall simulation (minimum 5)")
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generate_samples(args.model_dir, args.model_type, args.n_gen, args.years, device)