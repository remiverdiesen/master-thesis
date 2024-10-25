import numpy as np
import torch
import os
import logging
from torch.nn.functional import softmax
from scipy.stats import genextreme, poisson

# Adjusted for seasonal generation
EXPERIMENT = 2
model_type = 'GPD'
seasons = ['DJF', 'MAM', 'JJA', 'SON']
season_length = {'DJF': 90*24*12, 'MAM': 92*24*12, 'JJA': 92*24*12, 'SON': 91*24*12}  # Each value represents number of 5-min intervals

# Load the specific model for the season
model_path = f"experiments{EXPERIMENT}\\{season}\\model\\{model_type}-GAN\\netG_final.pth"


def generate_seasonal_data(season, config, data_handler, num_samples, netG):
    logger.info(f"Generating data for season: {season}")

    # Load the specific model for the season
    model_path = f"{config.models_dir}\\{season}-GAN\\netG_final.pth"
    netG.load_state_dict(torch.load(model_path, weights_only=True, map_location=config.device))
    netG.eval()

    # Generate samples for the current season
    U_samples = []
    batch_size = config.batch_size
    noise_dim = config.noise_dim

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - i)
            noise = torch.randn(current_batch_size, noise_dim, device=config.device)
            U_ = netG(noise)
            U_samples.append(U_.cpu().numpy())

    # Concatenate all generated samples
    U_samples = np.concatenate(U_samples, axis=0)
    logger.info(f"Generated {U_samples.shape[0]} samples for {season}.")

    # Remove padding if necessary
    pad = 1
    U_samples = U_samples[:, :, pad:-pad, pad:-pad]
    U_samples = np.squeeze(U_samples, axis=1)

    # Normalize to (0, 1) if required
    U_samples = (U_samples + 1) / 2  

    # Transform back to original scale using the inverse GEV/GPD
    Z_generated = inverse_transform(U_samples, data_handler.Z_train, data_handler.params, data_handler.ids_, config)
    logger.info(f"Transformed generated samples for {season} back to original scale.")

    return Z_generated


def poisson_sample_timing(season_length, avg_events):
    # Generate the number of events using Poisson distribution
    num_events = np.random.poisson(lam=avg_events)
    
    # Randomly assign event timings within the season length
    event_times = np.sort(np.random.uniform(0, season_length, num_events).astype(int))
    return event_times


def generate_yearly_data(config, data_handler, netG, num_years):
    yearly_samples = []

    for _ in range(num_years):
        yearly_data = []
        
        for season in seasons:
            num_samples = season_length[season]  # Number of samples to generate for each season
            Z_generated = generate_seasonal_data(season, config, data_handler, num_samples, netG)
            
            # Determine the average number of events for the season
            avg_events = config.avg_events_per_season[season]
            event_times = poisson_sample_timing(season_length[season], avg_events)
            
            # Assign precipitation values based on event timings
            seasonal_data = np.zeros(season_length[season])
            seasonal_data[event_times] = Z_generated[:len(event_times)]
            
            yearly_data.append(seasonal_data)
        
        # Concatenate all seasonal data to form a full year
        yearly_samples.append(np.concatenate(yearly_data))
    
    return np.array(yearly_samples)

def evaluate_yearly_data(yearly_generated_data, yearly_real_data, config, netG):
    logger.info("Evaluating the generated yearly data...")

    # Use your existing evaluation functions
    ks_result, ks_list = ks_test(yearly_generated_data, yearly_real_data)
    logger.info(f"KS Test result:       {round(ks_result, 4)}")

    chi_result = chi_statistic(yearly_generated_data, yearly_real_data)
    logger.info(f"Chi-statistic:        {round(chi_result, 4)}")

    mse = mean_squared_error(yearly_generated_data, yearly_real_data)
    logger.info(f"Mean Squared Error:   {round(mse, 4)}")

    fid = frechet_inception_distance(yearly_generated_data, yearly_real_data)
    logger.info(f"Frechet Distance:     {round(fid, 4)}")

    score = inception_score(yearly_generated_data, netG, config)
    logger.info(f"Inception Score:      {round(score, 4)}")

    wasserstein = wasserstein_distance_grid(yearly_generated_data, yearly_real_data)
    logger.info(f"Wasserstein Distance: {round(wasserstein, 4)}")
