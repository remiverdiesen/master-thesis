import os
import yaml
import pandas as pd
from evt_utils import fit_gev
from data_utils import load_and_preprocess_data


def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

# Load configuration
config = load_config("config/config.yaml")

# Load and preprocess data (get train_data, discarding NaN)
_, train_data = load_and_preprocess_data(config)
print(f"Using {train_data.shape[0]} clean samples for GEV fitting.")

# Fit GEV parameters
params = fit_gev(train_data) 

# Flatten for CSV saving
params_flat = params.reshape(18 * 22, 3)
params_df = pd.DataFrame(params_flat, columns=['c', 'loc', 'scale'])

# Ensure directory exists
os.makedirs('data', exist_ok=True)

# Save to CSV
params_df.to_csv('data/params.csv', index=False)
print("GEV parameters saved to data/params.csv")