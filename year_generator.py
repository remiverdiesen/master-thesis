# Assuming you've generated data for each season (DJF, MAM, JJA, SON) separately
# Load or generate seasonal data:
seasonal_data = {
    'DJF': Z_generated_djf,  # Replace with your generated data for each season
    'MAM': Z_generated_mam,
    'JJA': Z_generated_jja,
    'SON': Z_generated_son
}

# Concatenate the seasonal data to form a single yearly sample
yearly_sample = np.concatenate([
    seasonal_data['DJF'], 
    seasonal_data['MAM'], 
    seasonal_data['JJA'], 
    seasonal_data['SON']
], axis=0)

logger.info(f"Successfully created a yearly sample with shape: {yearly_sample.shape}")

# Load weights and generate seasonal data
def generate_season_data(season_model_path, num_samples, noise_dim, params, model_type='GEV'):
    # Load model weights
    netG = Generator(noise_dim=noise_dim)
    netG.load_state_dict(torch.load(season_model_path, map_location=device))
    netG.to(device)
    netG.eval()
    
    # Generate samples
    U_samples = []
    with torch.no_grad():
        for _ in range(num_samples // batch_size):
            noise = torch.randn(batch_size, noise_dim, device=device)
            generated = netG(noise).cpu().numpy()
            U_samples.append(generated)
    
    # Concatenate and transform back
    U_samples = np.concatenate(U_samples, axis=0)
    Z_generated = transform_samples(U_samples, params, model_type)
    
    return Z_generated

# Example generating for each season
djf_data = generate_season_data('path_to_djf_model.pth', 10000, 100, params_djf, model_type='GEV')
mam_data = generate_season_data('path_to_mam_model.pth', 10000, 100, params_mam, model_type='GEV')
jja_data = generate_season_data('path_to_jja_model.pth', 10000, 100, params_jja, model_type='GEV')
son_data = generate_season_data('path_to_son_model.pth', 10000, 100, params_son, model_type='GEV')

# Combine to yearly sample
yearly_sample = np.concatenate([djf_data, mam_data, jja_data, son_data], axis=0)
logger.info(f"Generated a complete yearly sample of shape: {yearly_sample.shape}")


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
