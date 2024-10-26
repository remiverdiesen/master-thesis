import os
import logging
import torch
import numpy as np
print("CUDA Available: ", torch.cuda.is_available())
from config import Config
from data_handler import DataHandler
from models import Generator
from utils import inverse_transform

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(lineno)d - %(levelname)s - %(message)s ")

def generate():
    logger.info("\n\n\n #####################\n\n\n Generating samples...\n")

    EXPERIMENT = 1 
    file_path = f"experiments\\1\\config.txt"	


    config = Config.load_config(train=False, file_path=file_path)

    # Initialize DataHandler
    data_handler = DataHandler(config)

    # Check if CUDA is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained Generator
    netG = Generator(config.noise_dim, batch_norm=config.batch_norm).to(device)
    model_type = config.model_type

    # Define model path based on experiment
    model_path = os.path.join(config.models_dir, f'{model_type}-GAN', 'netG_final.pth')
    
    try:
        netG.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
        netG.eval()
        logger.info(f"Loaded the trained Generator for {model_type} model.\n")
    except Exception as e:
        logger.error(f"Error loading the Generator model: {e}")
        return

    # Configuration for sample generation
    num_samples = 1000
    batch_size = config.batch_size
    noise_dim = config.noise_dim

    # Create folder to store generated samples based on the training period
    output_dir = os.path.join(config.data_dir, f'generated_samples_{model_type}')
    os.makedirs(output_dir, exist_ok=True)

    U_samples = []

   
    netG.to(device)  # Move the generator model to the selected device

    # Generate n new data points from the trained Generator with uniform margins
    logger.info(f"Start generating {num_samples} samples...")
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - i)
            
            # Generate noise tensor directly on the correct device
            noise = torch.randn(current_batch_size, noise_dim, device=device)
            
            # Generate samples using the Generator on the device
            U_ = netG(noise)
            
            # Move samples back to CPU and append them to the list
            U_samples.append(U_.cpu().numpy())

    # Concatenate all generated samples
    U_samples = np.concatenate(U_samples, axis=0)
    logger.info(f"Successfully generated {U_samples.shape[0]} samples!")

    # Optional: Remove padding from samples if applicable
    pad = 1  # Adjust if padding is different
    U_samples = U_samples[:, :, pad:-pad, pad:-pad]

    # Squeeze the channel dimension
    U_samples = np.squeeze(U_samples, axis=1)

    # Normalize to (0, 1) if necessary 
    U_samples = (U_samples + 1) / 2  
    logger.debug(f"After Normalization: U_samples min: {U_samples.min()}, max: {U_samples.max()}") 

    # Transform generated samples back to original scale using inverse GEV/GPD CDFs 
    Z_generated = inverse_transform(U_samples, data_handler.Z_train, data_handler.params, config)

    logger.debug(f"Generated samples shape: {Z_generated.shape}")

    # Store the generated samples
    period_generated_path = os.path.join(config.results_dir, f'{model_type}_generated_samples.npy')
    
    np.save(period_generated_path, Z_generated)


    logger.info("All samples successfully generated and stored.")

if __name__ == "__main__":
    generate()
