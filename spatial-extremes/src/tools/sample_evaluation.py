import numpy as np
import scipy.stats
from scipy.stats import ks_2samp, wasserstein_distance, pearsonr, spearmanr, skew, kurtosis
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import probplot
import warnings
import random

# ------------------ Helper Functions ------------------

def select_random_grid_points(grid_shape, num_points=10):
    """
    Select random grid points from a given grid shape.
    
    Parameters:
    - grid_shape: Tuple representing the shape of the grid (height, width).
    - num_points: Number of random points to select.
    
    Returns:
    - List of tuples representing random (latitude, longitude) indices.
    """
    height, width = grid_shape
    selected_points = [(random.randint(0, height - 1), random.randint(0, width - 1)) for _ in range(num_points)]
    return selected_points

def aggregate_sample(sample):
    """
    Aggregate a sample by averaging over the grid points (18x22) for each time step.
    
    Parameters:
    - sample: 3D array of shape (time_steps, grid_height, grid_width)
    
    Returns:
    - aggregated: 1D array of mean values across the grid for each time step.
    """
    aggregated = np.mean(sample, axis=(1, 2))  # Reduces (1000, 18, 22) to (1000,)
    return aggregated

# ------------------ Evaluation Metrics Functions ------------------

def ks_test(sample1, sample2):
    """
    Perform the Kolmogorov-Smirnov test between two data samples.
    
    Parameters:
    - sample1: Array of precipitation values from the first sample.
    - sample2: Array of precipitation values from the second sample.
    
    Returns:
    - ks_stat: KS statistic.
    - p_value: p-value of the KS test.
    """
    ks_stat, p_value = ks_2samp(sample1, sample2)
    return ks_stat, p_value

def wasserstein(sample1, sample2):
    """
    Calculate the Wasserstein distance between two data samples.
    
    Parameters:
    - sample1: Array of precipitation values from the first sample.
    - sample2: Array of precipitation values from the second sample.
    
    Returns:
    - distance: Computed Wasserstein distance.
    """
    return wasserstein_distance(sample1, sample2)

def rmse(sample1, sample2):
    """
    Calculate the Root Mean Square Error (RMSE) between two data samples.
    
    Parameters:
    - sample1: Array of precipitation values from the first sample.
    - sample2: Array of precipitation values from the second sample.
    
    Returns:
    - rmse_value: Computed RMSE.
    """
    return np.sqrt(np.mean((sample1 - sample2) ** 2))

def mae(sample1, sample2):
    """
    Calculate the Mean Absolute Error (MAE) between two data samples.
    
    Parameters:
    - sample1: Array of precipitation values from the first sample.
    - sample2: Array of precipitation values from the second sample.
    
    Returns:
    - mae_value: Computed MAE.
    """
    return np.mean(np.abs(sample1 - sample2))

def r_squared(sample1, sample2):
    """
    Calculate the coefficient of determination (R^2) between two data samples.
    
    Parameters:
    - sample1: Array of precipitation values from the first sample.
    - sample2: Array of precipitation values from the second sample.
    
    Returns:
    - r_squared_value: Computed R^2 value.
    """
    correlation_matrix = np.corrcoef(sample1, sample2)
    correlation_xy = correlation_matrix[0, 1]
    return correlation_xy ** 2

# ------------------ Visualization Functions ------------------

def qq_plot(sample1, sample2, title='Q-Q Plot'):
    """
    Generate a Q-Q plot comparing two data samples.
    
    Parameters:
    - sample1: Array of precipitation values from the first sample.
    - sample2: Array of precipitation values from the second sample.
    - title: Title of the Q-Q plot.
    
    Returns:
    - None (displays the plot).
    """
    plt.figure(figsize=(8, 6))
    probplot(sample1, dist="norm", plot=plt)
    probplot(sample2, dist="norm", plot=plt)
    plt.title(title)
    plt.xlabel('Quantiles')
    plt.ylabel('Precipitation')
    plt.grid(True)
    plt.legend(['Sample 1', 'Sample 2'])
    plt.show()

def kde_plot(sample1, sample2, sample3, title='KDE Plot'):
    """
    Generate a Kernel Density Estimation (KDE) plot comparing three data samples.
    
    Parameters:
    - sample1, sample2, sample3: Arrays of precipitation values from each sample.
    - title: Title of the KDE plot.
    
    Returns:
    - None (displays the plot).
    """
    plt.figure(figsize=(10, 8))
    sns.kdeplot(sample1, label='Sample 1', fill=True, alpha=0.4)
    sns.kdeplot(sample2, label='Sample 2', fill=True, alpha=0.4)
    sns.kdeplot(sample3, label='Sample 3', fill=True, alpha=0.4)
    plt.title(title)
    plt.xlabel('Precipitation')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

# ------------------ Evaluation and Comparison Functions ------------------

def evaluate_at_random_points(sample1, sample2, sample3, grid_shape, num_points=10):
    """
    Evaluate multiple metrics at random grid points.
    
    Parameters:
    - sample1, sample2, sample3: 3D arrays of generated precipitation values.
    - grid_shape: Shape of the grid (e.g., (18, 22)).
    - num_points: Number of random grid points to select.
    
    Returns:
    - pointwise_metrics: Dictionary with metrics at random points.
    """
    random_points = select_random_grid_points(grid_shape, num_points)
    pointwise_metrics = {}
    
    for idx, (i, j) in enumerate(random_points):
        metrics = {}
        s1_values = sample1[:, i, j]
        s2_values = sample2[:, i, j]
        s3_values = sample3[:, i, j]
        
        metrics['KS Test (S1 vs S2)'] = ks_test(s1_values, s2_values)
        metrics['KS Test (S1 vs S3)'] = ks_test(s1_values, s3_values)
        metrics['KS Test (S2 vs S3)'] = ks_test(s2_values, s3_values)
        
        metrics['Wasserstein (S1 vs S2)'] = wasserstein(s1_values, s2_values)
        metrics['Wasserstein (S1 vs S3)'] = wasserstein(s1_values, s3_values)
        metrics['Wasserstein (S2 vs S3)'] = wasserstein(s2_values, s3_values)
        
        metrics['RMSE (S1 vs S2)'] = rmse(s1_values, s2_values)
        metrics['RMSE (S1 vs S3)'] = rmse(s1_values, s3_values)
        metrics['RMSE (S2 vs S3)'] = rmse(s2_values, s3_values)

        metrics['MAE (S1 vs S2)'] = mae(s1_values, s2_values)
        metrics['MAE (S1 vs S3)'] = mae(s1_values, s3_values)
        metrics['MAE (S2 vs S3)'] = mae(s2_values, s3_values)
        
        metrics['R^2 (S1 vs S2)'] = r_squared(s1_values, s2_values)
        metrics['R^2 (S1 vs S3)'] = r_squared(s1_values, s3_values)
        metrics['R^2 (S2 vs S3)'] = r_squared(s2_values, s3_values)
        
        pointwise_metrics[f'Point {idx + 1} (Grid {i}, {j})'] = metrics
    
    return pointwise_metrics

def grid_wide_comparison(sample1, sample2, sample3):
    """
    Perform grid-wide comparisons using KDE, Q-Q plots, MAE, and R^2 metrics.
    
    Parameters:
    - sample1, sample2, sample3: 3D arrays of generated precipitation values.
    
    Returns:
    - grid_metrics: Dictionary with aggregated metrics across the grid.
    """
    # Aggregate the data for grid-wide comparisons
    sample1_aggregated = aggregate_sample(sample1)
    sample2_aggregated = aggregate_sample(sample2)
    sample3_aggregated = aggregate_sample(sample3)
    
    # Generate KDE and Q-Q plots
    kde_plot(sample1_aggregated, sample2_aggregated, sample3_aggregated, title='KDE Plot: Grid-Wide')
    qq_plot(sample1_aggregated, sample2_aggregated, title='Q-Q Plot: Sample 1 vs Sample 2')
    qq_plot(sample1_aggregated, sample3_aggregated, title='Q-Q Plot: Sample 1 vs Sample 3')
    qq_plot(sample2_aggregated, sample3_aggregated, title='Q-Q Plot: Sample 2 vs Sample 3')
    
    # Calculate metrics
    grid_metrics = {
        'MAE (S1 vs S2)': mae(sample1_aggregated, sample2_aggregated),
        'MAE (S1 vs S3)': mae(sample1_aggregated, sample3_aggregated),
        'MAE (S2 vs S3)': mae(sample2_aggregated, sample3_aggregated),
        'R^2 (S1 vs S2)': r_squared(sample1_aggregated, sample2_aggregated),
        'R^2 (S1 vs S3)': r_squared(sample1_aggregated, sample3_aggregated),
        'R^2 (S2 vs S3)': r_squared(sample2_aggregated, sample3_aggregated)
    }
    
    return grid_metrics

# ------------------ Main Script ------------------

# Load samples
sample_eGPD = np.load('experiments/1/results/eGPD_generated_samples.npy')
sample_GPD = np.load('experiments/1/results/GPD_generated_samples.npy')
sample_GEV = np.load('experiments/1/results/GEV_generated_samples.npy')

# Perform point-wise comparisons
pointwise_metrics = evaluate_at_random_points(sample_eGPD, sample_GPD, sample_GEV, grid_shape=(18, 22))

# Perform grid-wide comparisons
grid_metrics = grid_wide_comparison(sample_eGPD, sample_GPD, sample_GEV)

# Print the grid-wide metrics
print("Grid-wide Aggregated Metrics:")
for metric, value in grid_metrics.items():
    print(f"{metric}: {value}")

# Print the point-wise metrics
print("\nPoint-wise Metrics at Random Grid Points:")
for point, metrics in pointwise_metrics.items():
    print(f"{point}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value}")


def mean_residual_life_plot(data, thresholds):
    means = []
    for t in thresholds:
        exceedances = data[data > t] - t
        if len(exceedances) > 0:
            means.append(np.mean(exceedances))
        else:
            means.append(np.nan)

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, means, marker='o')
    plt.xlabel('Threshold')
    plt.ylabel('Mean Excess')
    plt.title('Mean Residual Life Plot')
    plt.grid(True)
    plt.show()

# Example usage
data = np.random.randn(1000)  # Replace with your dataset
thresholds = np.linspace(np.min(data), np.percentile(data, 95), 50)
mean_residual_life_plot(data, thresholds)