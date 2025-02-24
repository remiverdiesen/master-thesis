# src/settings.py
import os
import datetime
import logging
import numpy as np
from dataclasses import dataclass

@dataclass
class Grid:
    Nx: int           # Number of grid cells in x direction
    Ny: int           # Number of grid cells in y direction
    dx: float         # (m) Grid spacing in x direction
    dy: float         # (m) Grid spacing in y direction
    area: np.ndarray  # Bottom topography (m)
    Sx: int           # sample middle point X 
    Sy: int           # sample middle point Y 
    Sd: int           # sample dimension 

@dataclass
class Time:
    dt: float         # (s) Time step
    total: int        # (s) Total number of time steps
    sf: int           # (#) Sub samping factor 

@dataclass
class Solver:
    tolerance: float         # (s) Time step
    max_iteration: int      # (s) End time
    alpha: float             # Damping factor
    epsilon: float           # 

@dataclass
class Precipitation:
    enabled: bool
    area: np.ndarray
    rate: float
    t_start: float
    t_end: float

@dataclass
class Absorption:
    enabled: bool
    area: np.ndarray
    rate: float

@dataclass
class Flow_measure:
    enabled: bool
    area: np.ndarray
    dir: str            #  "x", "y", "z", "xy", "xyz" 

@dataclass
class Physical:
    g: float         # Acceleration due to gravity (m/s^2)
    nu: float        # Kinematic viscosity         (m^2/s)    
    m: float         # Manning's  coefficient      (s/m^(1/3))
    gamma_T: float   # Wind stress coefficient
    u_a: float       # Wind speed in x-direction
    v_a: float       # Wind speed in y-direction
    gamma:np.ndarray # Friction coefficient gamma (can be spatially varying)
    infiltration: float     # Infiltration rate (m/s)     
    flower: float    #  factor for flow tuning 

@dataclass
class Config:
    logging_level: str
    run_type: str
    max_plots: int
    result_dir: str
    date_time: str
    store_3D_plots: bool 
    images_dir: str
    mid_fs_update: bool
    bag: np.ndarray
    grid: Grid
    time: Time
    solver: Solver
    precipitation: Precipitation
    absorption: Absorption 
    flow_measure: Flow_measure
    physical: Physical

    def save_to_file(self):
        file_name = '__CONFIG.txt'
        file_path = os.path.join(self.result_dir, file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as f:
            f.write(f"Run Type:          {self.run_type} \n")
            f.write(f"Result Directory:  {self.result_dir} \n")
            f.write(f"Folder Name:       {self.date_time} \n")
            f.write(f"Maximum Plots:     {self.max_plots} \n")
            f.write(f"Store_3D_plots:    {self.store_3D_plots} \n")
            f.write(f"Images Directory:  {self.images_dir} \n")
            f.write(f"Mid_FlowSpeed_Upd: {self.mid_fs_update} \n\n")

            f.write(f"Grid:\n")
            f.write(f"   Nx: {self.grid.Nx} \n")
            f.write(f"   Ny: {self.grid.Ny} \n")
            f.write(f"   dx: {self.grid.dx} \n")
            f.write(f"   dy: {self.grid.dy} \n\n")
            
            f.write(f"Time:\n")
            f.write(f"   dt:    {self.time.dt} [s] \n")
            f.write(f"   total: {self.time.total} [s] \n")
            f.write(f"   sf:    {self.time.sf} (sub-sample factor) \n\n")

            # f.write(f"Solver:\n")
            # f.write(f"  tolerance:  {self.solver.tolerance}\n")
            # f.write(f"  max iters:  {self.solver.max_iteration}\n")
            # f.write(f"  Alpha:      {self.solver.alpha}\n")
            # f.write(f"  Epsilon:    {self.solver.epsilon}\n\n")
            
            f.write(f"Precipitation:\n")
            f.write(f"   Enabled:    {self.precipitation.enabled} \n")
            f.write(f"   Rate:       {self.precipitation.rate} [mm/hr] \n")
            f.write(f"   Start:      {self.precipitation.t_start} [s] \n")
            f.write(f"   End :       {self.precipitation.t_end} [s] \n\n")

            f.write(f"Absorption:\n")
            f.write(f"   Enabled:    {self.absorption.enabled} \n")
            f.write(f"   Rate:       {self.absorption.rate} [mm/hr] \n\n")

            f.write(f"Flow Rate:\n")
            f.write(f"   Enabled:          {self.flow_measure.enabled} \n")
            #  f.write(f"   Measurement Area: {self.flow_measure.area}\n")
            f.write(f"   Direction:        {self.flow_measure.dir} \n\n")

            f.write(f"Physical Parameters:\n")
            f.write(f"   Gravity:             {self.physical.g} [m/s2] \n")
            # f.write(f"  kinematic viscosity: {self.physical.nu}\n")
            # f.write(f"  Manning Coefficient: {self.physical.m}\n")
            # f.write(f"  wind stress coeff:   {self.physical.gamma_T}\n")
            # f.write(f"  wind speed (u):      {self.physical.u_a}\n")
            # f.write(f"  wind speed (v):      {self.physical.v_a}\n")
            # f.write(f"  friction coefficient (gamma): {self.physical.gamma}\n")
            f.write(f"   Infiltration Rate: {self.physical.infiltration}\n")
            f.write(f"   Flow Extra Rate: {self.physical.flower}\n\n")

@dataclass
class BAG:
    grid: np.ndarray # Nx,Ny int grid where 1 = building, 0 = not building

@dataclass
class BGT:
    grid: np.ndarray # Nx,Ny int grid where ...

@dataclass
class AHN:
    h: np.ndarray # Nx,Ny float grid with elevation data

def initialize_settings(config):
    return Config(
        logging_level=config['logging_level'],
        run_type=config['run_type'],
        max_plots = config['max_plots'],
        date_time = f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}',
        result_dir = os.path.join( os.getcwd(), 'results'),
        images_dir = os.path.join( os.getcwd(), 'images' ),
        bag = [0, 0],
        store_3D_plots=config['store_3D_plots'],
        mid_fs_update=config['mid_fs_update'],
        grid=Grid(**config['grid']),
        time=Time(**config['time']),
        precipitation=Precipitation(**config['precipitation']),
        absorption=Absorption(**config['absorption']), 
        flow_measure=Flow_measure(**config['flow_measure']), 
        solver=Solver(**config['solver']),
        physical=Physical(**config['physical'])
    )

def logger_setup(config):
    logging.basicConfig(level=getattr(logging, config.logging_level), 
                        format='%(message)s')
    #  format='%(asctime)s - %(levelname)s - %(message)s - '
    #                            'File: %(pathname)s - '
    #                            'Line: %(lineno)d - '
    #                            'Method: %(funcName)s')


    logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)
    logging.getLogger("PIL").setLevel(logging.WARNING)
# Output directory
out_dir = os.path.join(os.getcwd())




