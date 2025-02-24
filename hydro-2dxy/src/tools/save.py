import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, FuncNorm
from matplotlib.ticker import MultipleLocator

#  out_dir = os.path.join(os.getcwd())

def save_sim_results(config, timing_results):
    output_path = os.path.join(config.result_dir, f'method_performance.txt')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        with open(output_path, 'a') as f:
            json.dump(timing_results, f, indent=4)
            f.write('\n')
    except IOError as e:
        logging.error(f"Error writing to file: {e}")

def save_3D_plot(config, description: str, area: np.ndarray ):
    """
    Plots and saves the generated grid for specific areas/testcases (with type info)

    Args:
        config:   the set of runtime configuration conditions 
        text:     testcase description (free string)
        area:     A 2D array with shape (Ny, Nx, z) 
                        containing the x, y coordinates and values of the grid.

    Returns: 
        0

    Plots: 
        A 3D map image 
    """
    try: 
        Ty, Tx = area.shape
        dy, dx = config.grid.dy, config.grid.dx
        # Generate X and Y grid coordinates based on the shape of h
        x = np.linspace(Tx * dx, 0, Tx)
        # x = np.linspace(Tx * dx, 0, Tx)
        y = np.linspace(0, Ty * dy, Ty)
        # y = np.linspace(Ty * dy, 0, Ty)
        X, Y = np.meshgrid(x, y)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, area, cmap='terrain', edgecolor='k', linewidth=0.5)
        ax.view_init(elev=15, azim=150)  # Set elev to 10 deg and azimuth to 180 deg
        ax.set_xlabel("x ( distance in [m] )")
        ax.set_ylabel("y ( distance in [m] )")
        ax.set_zlabel("z ( elevation in [m] )")

        benchmark_number = config.run_type[-1]
        area_type = config.run_type[ :-1]
        # Construct the directory path
        if area_type == "benchmark" : 
            if benchmark_number == 0 : 
                tcd = f"{area_type}{benchmark_number} - R.P.J. Verdiesen, 2024"
            else :
                tcd = f"{area_type}{benchmark_number} - Di Giammarco, 1996"
            # save_dir = os.path.join(config.result_dir, 
            #     "benchmark", str(benchmark_number))
        else :
            tcd = f"Environment {config.run_type}" 
            # save_dir = os.path.join(config.result_dir, 
            #     "area", config.run_type, config.date_time)     

        ax.set_title(f"TestCase: {tcd}, {description} ") 

        # Save the figure as a PNG file in config.result_dir
        file_name = f"{description}_3D.png"
        output_path = os.path.join(config.result_dir, file_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, format='png')
        plt.close(fig)
    
    except Exception as reason:
       string = "An error occurred in function save_3D_plot !"
       logging.error(f"{string}\nException{reason}")

    return 0 


def save_2D_plot(config, description: str, area: np.ndarray ):
    """
    Plots and saves the generated grid for specific areas/testcases (with type info)

    Args:
        config:   the set of runtime configuration conditions 
        text:     testcase description (free string)
        area:     A 2D array with shape (Ny, Nx, z) 
                        containing the x, y coordinates and values of the grid.

    Returns: 
        0

    Plots: 
        A 2D map image 
    """
    try: 
        Ty, Tx = area.shape
        dy, dx = config.grid.dy, config.grid.dx

        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=150)

        plot_2D = ax.imshow(area, cmap='terrain' ) 
        cbar1 = fig.colorbar(plot_2D, ax=ax, orientation='vertical')
        cbar1.set_label("yes=1, no=0")

        ax.set_xlabel("x ( distance in [m] )")
        ax.set_ylabel("y ( distance in [m] )")

        benchmark_number = config.run_type[-1]
        area_type = config.run_type[ :-1]
        # Construct the directory path
        if area_type == "benchmark" : 
            if benchmark_number == 0 : 
                tcd = f"{area_type}{benchmark_number} - R.P.J. Verdiesen, 2024"
            else :
                tcd = f"{area_type}{benchmark_number} - Di Giammarco, 1996"
            # save_dir = os.path.join(config.result_dir, 
            #     "benchmark", str(benchmark_number))
        else :
            tcd = f"Environment {config.run_type}" 
            # save_dir = os.path.join(config.result_dir, 
            #     "area", config.run_type, config.date_time)     

        ax.set_title(f"Case: {tcd}, {description} ") 

        # Save the figure as a PNG file in config.result_dir
        file_name = f"{description}_2D.png"
        output_path = os.path.join(config.result_dir, file_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, format='png')
        plt.close(fig)
    
    except Exception as reason:
       string = "An error occurred in function save_2D_plot !"
       logging.error(f"{string}\nException{reason}")

    return 0 

def save_inundation_2D_plot(config, e: np.ndarray, dt: float, 
              time: float, pc: int, maxscale_e: float ) -> None:
    """
    Save a combined plot of the inundation level and flow speed as a single image.
    """

    try:
        # Create the figure with two subplots
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=150)

        lowest_value_allowed = 0
        highest_value_allowed = 0.5
        e = np.clip(e, lowest_value_allowed, highest_value_allowed)
        # Plot Inundation Level

        # funcnorm = FuncNorm((log2_transform, inverse_log2_transform), 
        #     vmin=lowest_value_allowed, vmax=maxscale_f)
        #  funcnorm = FuncNorm(vmin=lowest_value_allowed, vmax=maxscale_e)

        inundation_plot = ax.imshow(e, cmap='Blues' ) 
        cbar1 = fig.colorbar(inundation_plot, ax=ax, orientation='vertical')
        cbar1.set_label("Inundation [m]")
        ax.set_title(f"Inundation [{pc:03}] after {int(time // dt)} iterations/{time:.2f} seconds.")
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')

        # Save the combined plot as an image
        file_name = f"Inundation_Level_{pc:03}.png"
        output_path = os.path.join(config.result_dir, file_name )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, format='png')
        plt.close(fig)

    except Exception as reason:
       string = "An error occurred in function save_inundation_2D_plot() !"
       logging.error(f"{string}\nException{reason}")

def save_flow_velocity_2D_plot(config, f, dt: float, time: float, 
        pc: int, maxscale_f: float ) -> None:
    """
    Save a combined plot of the inundation level and flow speed as a single image.
    """

    try:
        # Create the figure with two subplots
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=150)

        #  Force minimal value to 
        lowest_value_allowed = 0
        highest_value_allowed = 0.5 
        f = np.clip(f, lowest_value_allowed, highest_value_allowed)
        # Plot Inundation Level

        # funcnorm = FuncNorm((log2_transform, inverse_log2_transform), 
        #     vmin=lowest_value_allowed, vmax=maxscale_f)
        
        # funcnorm = FuncNorm(vmin=lowest_value_allowed, vmax=maxscale_f)

        flow_velocity_plot = ax.imshow(f, cmap='Reds') 
        cbar1 = fig.colorbar(flow_velocity_plot, ax=ax, orientation='vertical')
        cbar1.set_label("Flow_Velocity [m/s]")
        ax.set_title(f"Flow_Velocity [{pc:03}] after {int(time // dt)} iterations/{time:.2f} seconds.")
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')

        # Save the combined plot as an image
        file_name = f"Flow_Velocity_2D_{pc:03}.png"
        output_path = os.path.join(config.result_dir, file_name )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, format='png')
        plt.close(fig)

    except Exception as reason:
       string = "An error occurred in function save_flow_velocity_2D_plot() !"
       logging.error(f"{string}\nException{reason}")


def save_flow_1D_plot(config, meas_type: str, 
                   pc: int, time_list: list, flow_list: list) -> None:
    """
    Save a plot pf Debiet (flux or flow-speed) as an image.
    """

    try:
        # Create the figure with two subplots
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        # Flow (m3/s) vs Time linewidth=2, markersize=12)
        # >>> plot(x, y, color='green', marker='o', linestyle='dashed'
        time_list_minutes = [t / 60 for t in time_list]
        
        ax.plot(time_list_minutes, flow_list, color='red', markersize=0.5, 
            marker='.', linewidth=0.5, linestyle='-')
        ax.set_title(f"Flow Rate ({meas_type}) [m3/s], plot {pc:03}")
        ax.set_xlabel('Time [minutes]')
        ax.set_ylabel('Flow Rate [m3/s]')

        # Set x-axis limits to start from zero to avoid negative numbers
        ax.set_xlim(0, None)
        # Set x-axis ticks to 5-minute intervals
        ax.xaxis.set_major_locator(MultipleLocator(5))

        # Rotate x-axis labels vertically
        ax.tick_params(axis='x', rotation=90)

        # Add horizontal gridlines according to y-axis ticks
        ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.7)
        
        # Expand the y-axis by 1 unit
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + 0.1)

        # Save the combined plot as an image
        file_name = f"Flow_Velocity_1D_{meas_type}_{pc:03}.png"
        output_path = os.path.join(config.result_dir, file_name )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, format='png')
        plt.close(fig)

    except Exception as reason:
       string = "An error occurred in function save_flow_1D_plot() !"
       logging.error(f"{string}\nException{reason}")

        # Define functions for log base 2 normalization

def log2_transform(x):
    return np.log2(x)

def inverse_log2_transform(x):
    return 2**x

