# Hydrolical Modelling

This repository contains a Python-based simulation tool for modeling shallow water dynamics on a 2D grid. It solves the shallow water equations to simulate water flow, surface elevation, and related phenomena like precipitation and absorption, suitable for applications such as flood modeling.

### Purpose

The code provides a framework to:

- Simulate water flow over varied topography (see figure).
- Compute surface elevation using numerical methods (e.g., Newton's method).
- Handle physical processes like precipitation, absorption, and obstacles.
- Visualize results through plots and logs.

![flow](hydro-2dxy\images\Flow_domain_and_computational_grid.pngGenerator.jpg)

### Key Components

- solver.py: Implements iterative solvers (Newton's method) to calculate surface elevation.
- compute.py: Contains functions for computing water depths, fluxes, and velocities.
- simulation.py: Runs the main simulation loop, integrating all components.
- settings.py: Defines configuration settings for grid, time, and physical parameters.
- tools/: Utilities for initialization, validation, and visualization.

### Requirements

- Python 3.x
- Libraries: numpy, scipy, matplotlib, xarray (optional for data loading)
  Install dependencies via:

### Basic Usage

1. Configure the Simulation:

   - Modify settings in settings.py or provide a custom config file.
   - Define grid size, time steps, and physical parameters (e.g., gravity, precipitation rate).

2. Run the Simulation:
   - Execute the main script (e.g., simulation.py):

Results are saved in the results/ directory (e.g., plots, logs). 3. Example Configuration: - Set run_type to "benchmark0" for a simple flat surface test. - Adjust grid.Nx, grid.Ny, and time.total for your domain and duration.

### Output

Plots: 2D/3D visualizations of water depth, flow velocity, etc., saved in images/.
Logs: Simulation progress and diagnostics in the console or log files.

### Notes

Designed for modularity; extend functionality by adding new solvers or physical processes.
Check config.py for detailed parameter options.
