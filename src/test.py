# src/test.py

import os
import logging
import torch
from torch.nn.functional import softmax
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genextreme, ks_2samp

from config import Config
from data_handler import DataHandler
from models import Generator
from utils import inverse_transform, ks_test, chi_statistic, mean_squared_error, frechet_inception_distance, inception_score, wasserstein_distance_grid

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(lineno)d - %(levelname)s - %(message)s ")

def test():
    logger.info("\n\n\n #####################\n\n\n Testing the trained Generator...\n")
    config = Config(train=False)

    # Initialize DataHandler
    data_handler = DataHandler(config)

    # Load the trained Generator
    netG = Generator(config.noise_dim, batch_norm=config.batch_norm).to(config.device)

    ###################################################################################################
    model_path = config.models_dir + '\\GEV-GAN\\netG_final.pth'





    ###################################################################################################

    netG.load_state_dict(torch.load(model_path, weights_only=True, map_location=config.device))
    netG.eval()
    logger.info(f"Loaded the trained Generator!")

    # Generate new samples NOTE use n_test to avoid size mismatch in CHi-aquared test, for actual modelling we can use any number of samples
    n_test = data_handler.Z_test.shape[0] 
    num_samples = n_test  
    batch_size = config.batch_size 
    noise_dim  = config.noise_dim  

    U_samples = []

    # Generate n new data points from the trained Generator with uniform margins
    logger.info(f"Start generating {num_samples} samples from trained Generator...")
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - i)
            noise = torch.randn(current_batch_size, noise_dim, device=config.device)
            U_ = netG(noise)
            U_samples.append(U_.cpu().numpy())

    logger.debug(f"Last generated sample shape: {U_.shape}") 

    # Concatenate all generated samples
    U_samples = np.concatenate(U_samples, axis=0)
    logger.info(f"Succesfully generated {U_samples.shape[0]} samples!")
    logger.debug(f"U_samples shape: {U_samples.shape}") 
      
    pad = 1  # Adjust if padding is different	
    
    # Remove padding from samples
    U_samples = U_samples[:, :, pad:-pad, pad:-pad]
    logger.debug(f"Removed padding from samples. New shape:   {U_samples.shape}") 

    # Squeeze the channel dimension
    U_samples = np.squeeze(U_samples, axis=1)
    logger.debug(f"Squeezed the channel dimension. New shape: {U_samples.shape}") 
    logger.debug(f"TEST: first generated sample: {U_samples[:, 0, 0]} ")          
    logger.debug(f"U_samples min: {U_samples.min()}, max: {U_samples.max()}")     

    # Normalize to (0, 1) if necessary 
    U_samples = (U_samples + 1) / 2  
    logger.debug(f"After Normalization: U_samples min: {U_samples.min()}, max: {U_samples.max()}") 

    # Transform generated samples back to original scale using inverse GEV\GPD CDFs OR empirical CDF
    Z_generated = inverse_transform(U_samples, data_handler.Z_train, data_handler.params, data_handler.ids_, config)
    logger.info("Transformed generated samples back to original scale.")

    assert Z_generated.shape == U_samples.shape, "Shape mismatch between Z_generated and U_samples"
    assert Z_generated.shape[1] == data_handler.Z_test.shape[1] and Z_generated.shape[2] == data_handler.Z_test.shape[2] , "Feature dimension mismatch"

    logger.debug(f"Z_generated shape: {Z_generated.shape}")
    logger.debug(f"Z_test shape: {data_handler.Z_test.shape}")


    evaluate_generated_samples(Z_generated, data_handler.Z_test.cpu().numpy(), data_handler.ids_, data_handler.GEV_params, config, netG)
    logger.info("Evaluation completed!")


def evaluate_generated_samples(Z_generated: np.ndarray, Z_test: np.ndarray, ids_: np.ndarray, 
                               GEV_params: np.ndarray, config: Config, netG: Generator):
    """
    Evaluate the generated samples by comparing with real data.
    """
    # Assuming real_data is already in the same shape and without padding
    logger.debug(f"Z_test shape:       {Z_test.shape}")
    logger.debug(f"Z_generated shape:  {Z_generated.shape}")
    logger.debug(f"params shape:       {params.shape}")
    logger.debug(f"ids_ shape:         {ids_.shape}")

    # Create directories for evaluation results
    evaluation_dir = config.evaluation_dir
    if not os.path.exists(evaluation_dir):
        os.makedirs(evaluation_dir)
    evaluation_plots_dir = os.path.join(evaluation_dir, 'evaluation_plots')
    if not os.path.exists(evaluation_plots_dir):
        os.makedirs(evaluation_plots_dir)

    # 1. Marginal distributions using KS test
    ks_result, ks_list = ks_test(Z_generated, Z_test)
    logger.info(f"KS Test result:       {round(ks_result, 4)}") # Goal = 0

    # 2.  Evaluating Dependence Structure (Chi-statistic for extremal dependence)
    chi_result = chi_statistic(Z_generated, Z_test)
    logger.info(f"Chi-statistic:        {round(chi_result, 4)}") # Goal = 1, perfectly captures the extremal dependence of the real data.

    # 3. Mean squared error
    mse = mean_squared_error(Z_generated, Z_test)
    logger.info(f"Mean Squared Error:   {round(mse, 4)}")       # Goal = 0

    # 4. Frechet Inception Distance
    fid = frechet_inception_distance(Z_generated, Z_test)       # Goal = 0, generated and real data have the same mean and covariance
    logger.info(f"Frechet Distance:     {round(fid, 4)}")

    # 5. Inception Score
    score = inception_score(Z_generated, netG, config)
    logger.info(f"Inception Score:      {round(score, 4)}")     # High Inception Score would imply that the generator is producing a wide range of realistic extreme events.

    # 6. Wasserstein Distance
    wasserstein = wasserstein_distance_grid(Z_generated, Z_test) #Goal = 0,  Generated and real data distributions are identical.
    logger.info(f"Wasserstein Distance: {round(wasserstein, 4)}")
    # give the rounden on 4 decimals
    
    # Visual inspection
    num_plots = 5
    for i in range(num_plots):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(Z_generated[i], cmap='viridis')
        plt.title(f'Generated Sample {i+1}')
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.imshow(Z_test[i % Z_test.shape[0]], cmap='viridis')
        plt.title(f'Real Sample {i+1}')
        plt.colorbar()
        plt.savefig(os.path.join(evaluation_plots_dir, f'sample_comparison_{i+1}.png'))
        plt.close()

if __name__ == "__main__":
    test()
