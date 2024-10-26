import os
import torch
import logging
import numpy as np
from src.config import Config
from src.models import Generator

# =============================================================================
# Set up logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - line: %(lineno)d - %(message)s "
)
logger = logging.getLogger(__name__)

# =============================================================================
# Load Configuration from Text File
# =============================================================================

logger.info(f"\n\n      Reading configuration..")
# Load the configuration
config_path = r"C:\Users\reverd\OneDrive - SAS\Documents\Thesis\3. Code\2. Modelling\Hazard\spatial-extremes\experiments\1\config.txt"
config = Config(config_path)
config = Config(train=False)

logger.info(f"\n\n      REad configuration from: {config_path}")
# Example usage:


# Generate 10,000 samples with a noise dimension
num_samples = 10000
batch_size = config.batch_size  
noise_dim = config.noise_dim  
models_dir = config.models_dir
model_type = config.model_type
params_file_path = config.params_file_path
ids_file_path = config.ids_file
use_empirical_cdf = config.use_empirical_cdf   # False 
model_path = os.path.join(models_dir,f'{model_type}-GAN', f"netG_final.pth")


# Load the necessary parameters for each season
params = np.load(params_file_path)  
ids = np.load(ids_file_path)

logger.info(f"Generating {num_samples} samples using the trained Generator for {model_type} model.")
logger.info(f"Model path:       {model_path}")
logger.info(f"Params file path: {params_file_path}")
logger.info(f"Batch size:       {batch_size}\n")



exit()

# =============================================================================
# Load the Generator Model and Generate Samples
# =============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

netG = Generator(noise_dim=noise_dim, batch_norm=True)  # Replace with actual model parameters
netG.load_state_dict(torch.load(model_path, map_location=device))
netG.to(device)
netG.eval()

# Log information about the model
logger.info(f"Loaded trained Generator for model: {model_path}")



# =============================================================================
# Generate Samples
# =============================================================================

U_samples = []

logger.info(f"Generating {num_samples} samples ...")
# Generate samples in batches
with torch.no_grad():
    for i in range(0, num_samples, batch_size):
        current_batch_size = min(batch_size, num_samples - i)
        noise = torch.randn(current_batch_size, noise_dim, device=device)
        generated = netG(noise).cpu().numpy()
        U_samples.append(generated)

    # Concatenate all generated samples
    U_samples = np.concatenate(U_samples, axis=0)
    logger.info(f"Generated {U_samples.shape[0]} samples.")

    # Remove padding if necessary
    pad = 1
    U_samples = U_samples[:, :, pad:-pad, pad:-pad]
    U_samples = np.squeeze(U_samples, axis=1)

    # Normalize to (0, 1) if required
    U_samples = (U_samples + 1) / 2  
    logger.info(f"Successfully generated {U_samples.shape[0]} samples with uniform margins.")


Z_train = np.zeros((1000, 100, 100))  # Replace with actual training data
Z_generated = inverse_transform(U_samples, Z_train, params, config)
logger.info("Transformed generated samples back to the original scale.")


print(f"Z_generated shape: {Z_generated.shape}")
print(f"Z_generated min: {Z_generated.min()}, max: {Z_generated.max()}")
print(f"Z_generated mean: {Z_generated.mean()}, std: {Z_generated.std()}")
print(f"Z_generated first sample: {Z_generated[0, 0, 0]}")
# =============================================================================
print("Done")
# =============================================================================


