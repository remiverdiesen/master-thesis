import torch
import numpy as np



seasons = ['DJF', 'MAM', 'JJA', 'SON']
EXPERIMENT = 1


file_path_config = r"file_to_config.txt"
file_path_param = r"file_to_params.txt"
model_path = 'path_to_your_trained_model.pth'  # Replace with the specific path to the trained seasonal model



config_file = read_file()

# Load the trained Generator
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Initialize your generator model
# Ensure you have the exact architecture that was used to train the model
netG = Generator(noise_dim=100, batch_norm=True)  # Replace with actual model parameters
netG.load_state_dict(torch.load(model_path, map_location=device))
netG.to(device)
netG.eval()

# Log information about the model
logger.info(f"Loaded trained Generator for season model: {model_path}")

# Generate 10,000 samples with a noise dimension
num_samples = 10000
batch_size = 128  # Define an appropriate batch size
noise_dim = 100  # Replace with the noise dimension used during training

U_samples = []

# Generate samples in batches
with torch.no_grad():
    for i in range(0, num_samples, batch_size):
        current_batch_size = min(batch_size, num_samples - i)
        noise = torch.randn(current_batch_size, noise_dim, device=device)
        generated = netG(noise).cpu().numpy()
        U_samples.append(generated)

# Concatenate all generated samples
U_samples = np.concatenate(U_samples, axis=0)
logger.info(f"Successfully generated {U_samples.shape[0]} samples with uniform margins.")


# Assume params contains the GEV or GPD parameters for each grid point (latitude, longitude)
# params.shape should be (n_lat, n_lon, 3), containing parameters for the inverse transformation

# Load the necessary parameters for each season
params = np.load(file_path_param)  # Replace with path to parameter file
use_empirical_cdf = False  # Assume we want to use GEV/GPD transformations

def transform_samples(U_samples, params, model_type):
    num_samples, n_lat, n_lon = U_samples.shape
    Z_generated = np.zeros_like(U_samples)
    
    # Apply the inverse transformation
    if model_type == 'GEV':
        for i in range(n_lat):
            for j in range(n_lon):
                Z_generated[:, i, j], _, _, _, _ = inverse_gev_(U_samples[:, i, j], params[i, j, :])
    elif model_type == 'GPD':
        for i in range(n_lat):
            for j in range(n_lon):
                Z_generated[:, i, j], _, _, _, _ = inverse_gpd_(U_samples[:, i, j], params[i, j, :])
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return Z_generated

# Example usage:
model_type = 'GEV'  # or 'GPD', based on your scenario
Z_generated = transform_samples(U_samples, params, model_type)
logger.info("Transformed generated samples back to the original scale.")
