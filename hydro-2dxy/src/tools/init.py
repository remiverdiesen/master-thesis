# src/utils/init.py
import os
import logging
import numpy as np
import xarray as xr
from ..config import Config, BAG, BGT, AHN
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from .save import save_3D_plot, save_2D_plot 

def generate_bottom_testcase(config: Config) -> np.ndarray:
    """
    Generates 4 grids for given benchmarks

    from "Verdiesen (and all 2024)", 
    - benchmark 0: flat surface with a drop of water flowing out

    and from "Di Giammarco 1996" , 
    - benchmark 1: slided surface 
    - benchmark 2: openbook testcase with slope in x direction
    - benchmark 3: openbook testcase with slope in x and y direction
    - benchmark 4: not implemented yet.

    Args:
        config:  the overall set of control data for the simulation 

    Returns:
        z:  np.array: ground level of the area under test 
        p:  np.array: percipitation area within the area under test 
        a:  np.array: absorbation area within the area under test
        f:  np.array: flow (measurement) area within the area under test
        
    """
    #   Schematic overview af the area's used below 
    #  
    #     x=800 m    20 m    x=800 m
    #  +------------+---+------------+
    #  |            | a |            | 
    #  |            | r |            | 
    #  |  area_A    | e |   area_C   |  Cy = 1000 m
    #  |            | a |            | 
    #  |            | _ |            |  
    #  |            | B |            |    Cells in Y : N_Ty = 324 
    #  +------------+---+------------+
    #  |                             |  
    #  |      drainage area_D        |  Dy = 620 m
    #  |                             |  
    #  +-----------------------------+
    #    Tx = Ax + Bx + Cx = 1620 m 
    #    Cells in X : N_Tx = 324 
    
    #  Define grid step sizes  
    dy, dx = config.grid.dy, config.grid.dx
    Sy, Sx, Sd = config.grid.Sy, config.grid.Sx, config.grid.Sd
    Ay, Ax = 1000, 800           # Area A dimensions
    By, Bx = Ay, 20              # Area B (central channel) dimensions
    Cy, Cx = Ay, Ax              # Area C dimensions
    Dy, Dx = 620, Ax + Bx + Cx   # Dy and Dx represent the Drain Area
    Ty, Tx = Ay + Dy, Dx         # Area T ( total ) dimensions

    dx2 = dx / 2 
    dy2 = dy / 2

    # Create grid arrays
    N_Ty, N_Tx = int(Ty / dy), int(Tx / dx)   # Number T cells (y, x direction)
    N_Ay, N_Ax = int(Ay / dy), int(Ax / dx)
    N_By, N_Bx = int(By / dy), int(Bx / dx)
    N_Cy, N_Cx = int(Cy / dy), int(Cx / dx)
    N_Dy, N_Dx = int(Dy / dy), int(Dx / dx)
    N_Dy2 = int( (Dy / dy) / 2) 

    H_Ty, H_Tx = 100, 100 
    #  Delta of the Slopes 
    d_Ax, d_Bx, d_Cx, d_Dx = 40, 20, 40,  0 
    d_Ay, d_By, d_Cy, d_Dy = 20, 20, 20,  12.8

    #  Slopes = Delta (in m) / Area_Size (x & y in m)
    S_Ax = d_Ax / Ax    #  40 / 800 = 0.05 
    S_Bx = 0.00 
    S_Cx = d_Cx / Cx    #  40 / 800 = 0.05 
    S_Ay = 0.00
    S_By = d_By / By
    S_Cy = 0.00
    S_Dy = d_Dy / Dy

    #  X slopes 
    A_Sx = np.linspace( 1-dx2/Ax,   dx2/Ax, N_Ax )
    B_Sx = np.linspace(        1,        1, N_Bx )
    C_Sx = np.linspace(   dx2/Cx, 1-dx2/Cx, N_Cx )

    #  Y slopes 
    A_Sy = np.linspace( 1, 1, N_Ay)
    B_Sy = np.linspace( 1 - dy2/By, dy2/By, N_By)
    C_Sy = np.linspace( 1, 1, N_Cy)
    D_Sy = np.linspace( dy2/Dy, 1 - dy2/Dy, N_Dy)

    #  config.precipitation.area = np.zeros((N_Ay, N_Ax))

    if config.run_type == "benchmark1":
        d_Bx, d_Cx = d_Dx, d_Dx
        B_Sy = np.linspace( 1, 1, N_By)

    if config.run_type == "benchmark2":
        d_Bx = d_Bx
        S_By = d_By / By  #  20 / 1000  = 0.02
        # dB_Sy = np.linspace( 1, dy2/By, N_By )

    if config.run_type == "benchmark3":
        S_Ay = d_Ay / Ay  #  20 / 1000  = 0.02
        S_By = d_By / By  #  20 / 1000  = 0.02
        S_Cy = d_Cy / Cy  #  20 / 1000  = 0.02
        # A_Sy = np.linspace( 1, 1 - S_Ay * Ay / d_Ax, N_Ay )
        A_Sy = np.linspace( 1, 1 - d_Ay / d_Ax, N_Ay)
        # C_Sy = np.linspace( 1, dy2/Cy, N_Cy )
        C_Sy = np.linspace( 1, 1 - d_Cy / d_Cx, N_Cy)

        # print(f"  S_Ay : {S_Ay}")
        # print(f"A_Sy (rel): \n{A_Sy}")

    # Define individual areas A, B, C, and D
    A = np.full((N_Ay, N_Ax),  d_Ax)
    B = np.full((N_By, N_Bx),  d_Bx)
    C = np.full((N_Cy, N_Cx),  d_Cx)
    D = np.full((N_Dy, N_Dx), -d_Bx)

    logging.debug(f"A.shape{A.shape}, B.shape{B.shape}, C.shape{C.shape}, D.size{D.shape}\n")

    if config.run_type == "benchmark0" :
        z = np.ones( ( config.grid.Ny, config.grid.Nx ) )  
        p = np.zeros( z.shape )
        a = np.zeros( z.shape )
        m = np.ones( z.shape )
        b = np.zeros( z.shape )
        #  m[Sy-1:Sy+1,Sx-1:Sx-1] = 1

    elif config.run_type == "benchmark1":
        d_Bx, d_Cx = d_Dx, d_Dx

        C = A * (A_Sx[None, :]) - d_Ax 
        A = A * (A_Sx[None, :]) 

        # Concatenate areas A, C in the X direction
        z = np.concatenate((A, C), axis=1)

        #   Precipitation Area is in A, 
        p = np.zeros( z.shape )
        p[ :N_Ay, : N_Ax] = 1           
        
        #   Absorbption Area is in D, only 
        a = np.zeros( z.shape )
        a[ :N_Ay, N_Ax : N_Ax + N_Cx] = 1  

        #  Measurement array 
        m = np.zeros( z.shape )
        m[ :N_Ay, N_Ax -2: N_Ax ] = 1 

        #  BAG (obstacles) array 
        b = np.zeros( z.shape )             # no road blocks 

    else : 
        # Apply slope to area A (left side slope)
        A = A * (A_Sx[None, :]) 
        A = A * (A_Sy[:, None]) 

        # Apply slope to area B (middel channel)
        B = B * (B_Sx[None, :]) 
        B = B * (B_Sy[:, None]) 
        B -= d_By 

        # Apply slope to area C (right side slope)
        C = C * (C_Sx[None, :]) 
        C = C * (C_Sy[:, None]) 

        D = D * (D_Sy[:, None]) - d_Bx 

        # Concatenate areas A, B, and C in the X direction
        ABC = np.concatenate((A, B, C), axis=1)

        # CBA = np.concatenate((C, B, A), axis=1)
        # Concatenate area D in the Y direction to ABC
        z = np.concatenate((ABC, D), axis=0)

        #   create "Walls" so slopes A and C can only 
        #   flow water down to area B (the river)
        #   z[N_Ay, :N_Ax ] = d_Ax
        #   z[N_By, N_Ax:N_Ax+N_Bx ] = -d_Bx
        #   z[N_Cy, N_Ax+N_Bx:N_Ax+N_Bx+N_Cx ] = d_Ax

        #  Bring entire test Area to a level guaranteed above 0 
        #  otherwise solvers will probably end "hanging" (without progress) 

        p = np.zeros( z.shape )
        p[ :N_Ay, :] = 1                   #   Precipitation Area is in A, B and C only 
        
        a = np.zeros( z.shape )
        #  now mask the area that needs absorption 
        a[ N_Ay: N_Ay + N_Dy, : N_Ax + N_Bx + N_Cx ] = 1  # absorption in D area only 
        # a[ N_Ay + 32: N_Ay + N_Dy, : N_Ax + N_Bx + N_Cx ] = 1  # absorption in D area only 

        m = np.zeros( z.shape )
        m[ N_Ay -2: N_Ay, N_Ax:N_Ax + N_Bx ] = 1   #  area B contact with with area D 

        b = np.zeros( z.shape )
        #  Small channel
        # b[ N_Ay:N_Ay + N_Dy , : N_Ax ] = 1 
        # b[ N_Ay:N_Ay + N_Dy, N_Ax + N_Bx : N_Ax + N_Bx + N_Cy ] = 1

        #  Entire Area D open field, only a roadblock to enter (through channel B)
        b[ N_Ay : N_Ay + 3, : N_Ax ] = 1 
        b[ N_Ay : N_Ay + 3, N_Ax + N_Bx : N_Ax + N_Bx + N_Cy ] = 1

    #  Lift the entire fround construction, such that
    #  the lowest point remains at 0 reference level 
    #  
    lowest_matrix_value = np.min( z )
    lift_matrix = 0 
    if ( lowest_matrix_value < 0 ) : 
        lift_matrix += -lowest_matrix_value
    z += lift_matrix

    #   logging.info(f"generate_bottom_testcase( config.result_dir :")
    #   logging.info(f"{config.result_dir}")
    
    return z, p, a, m, b

def set_simulation_type(config: Config) -> Config:  #  was str 

    benchmark_number = config.run_type[-1]
    benchmark_name = config.run_type[ : -1]
    if benchmark_name == "benchmark" : 

        config.result_dir = os.path.join(config.result_dir, 
            'benchmark', str(benchmark_number), config.date_time)
        os.makedirs(os.path.dirname(config.result_dir ), exist_ok=True)

        config.grid.dx = 5
        config.grid.dy = 5 
        z, p, a, m, b  = generate_bottom_testcase(config) 

        save_3D_plot(config, "_Ground_Levels", z)
        save_2D_plot(config, "_Precipitation_Area", p)
        save_2D_plot(config, "_Absorption_Area", a )
        save_2D_plot(config, "_Measurement_Area", m )
        save_2D_plot(config, "_Obstacles_BAG", b )

        config.grid.area = np.copy(z)
        config.precipitation.area = np.copy(p)
        config.absorption.area = np.copy(a)
        config.flow_measure.area = np.copy(m)
        config.bag = np.copy(b)

        config.time.dt = 1
        config.time.total = 3 * 60 * 60    
        config.precipitation.enabled = True
        config.precipitation.rate = 10.8   
        config.precipitation.t_start = 0
        config.precipitation.t_end = 60 * 90    # 90 minutes 
        config.absorption.enabled = False
        config.absorption.rate = 0
        config.flow_measure.enabled = True
        config.flow_measure.dir = "y"

        if config.run_type == "benchmark0":
            config.grid.Sy = config.grid.Ny // 2 + 2
            config.grid.Sx = config.grid.Nx // 2 + 2
            config.grid.Sd = 4
            config.grid.area = np.ones(z.shape)
            config.precipitation.enabled = False
            config.precipitation.rate = 0 
            config.precipitation.t_start = 0
            config.absorption.enabled = False
            config.absorption.rate = 1
            config.flow_measure.enabled = False
            config.flow_measure.dir = "y"

        elif config.run_type == "benchmark1" : 
            benchmark_number = config.run_type[-1]
            config.grid.Sx = 160 -1 
            config.grid.Sy = 100 -1  
            config.flow_measure.dir = "x"

        elif config.run_type == "benchmark2" : 
            config.grid.Sx = 160 + 2
            config.grid.Sy = 200     
        
        elif config.run_type == "benchmark3":
            config.grid.Sx = 160 + 2
            config.grid.Sy = 200     

        elif config.run_type == "benchmark4": 
            config.grid.Sx = 160 + 2
            config.grid.Sy = 200      
            string1 = "This benchmark is not implemented yet"
            string2 = "Please check the config file"
            raise ValueError(f"\n\\n     {string1}\n{string2}\n")

    elif ( len(config.run_type) > 0 ):
        
        grid_file = f"grid_05m_static_{config.run_type}.nc"
        local_data_set = os.path.join( os.getcwd(), "data", grid_file )

        #  local_data_set = f"r\"{grid_path}\"" 

        # if config.run_type == "Meerssen" : 
        #     ds = xr.open_dataset(r".\\data\\grid_05m_static_Meerssen.nc")

        # elif config.run_type == "Hilversum" : 
        #     ds = xr.open_dataset(r".\\data\\grid_05m_static_Hilversum.nc")

        # elif config.run_type == "Valkenburg" : 
        #     ds = xr.open_dataset(r".\\data\\grid_05m_static_Valkenburg.nc")

        # print(f"config.run_type = {config.run_type}") 
        # print(f"grid_file = {grid_file}") 
        # print(f"local_data_set = {local_data_set}") 
        # print(f"") 

        ds = xr.open_dataset( local_data_set )

        AHN = ds['AHN'].values
        BAG = ds['BAG'].values
             
        if config.run_type == "Bodegraven" : 
            z = AHN[ 3500 : 4500, 3500 : 4500 ]
            b = BAG[ 3500 : 4500, 3500 : 4500 ]

        elif config.run_type == "Valkenburg" : 
            # z = AHN[ 3600 : 4200, 3800 : 4400 ]
            # b = BAG[ 3600 : 4200, 3800 : 4400 ]
            z = AHN[ 3200 : 4200, 3500 : 4500 ]
            b = BAG[ 3500 : 4500, 3500 : 4500 ]

        elif config.run_type == "Meerssen" : 
            # z = AHN[ 3600 : 4200, 3800 : 4400 ]
            # b = BAG[ 3600 : 4200, 3800 : 4400 ]
            z = AHN[ 500 : 1500, 500 : 1500 ]
            b = BAG[ 500 : 1500, 500 : 1500  ]

        else : 
            z = AHN[ : ,  : ]
            b = BAG[ : ,  : ]


        string = f"AHN.shape: {AHN.shape} --> reduced --> z.shape: {z.shape}"
        logging.info(f"Info:    {string}") 

        string = f"BAG.shape: {BAG.shape} --> reduced --> b.shape: {b.shape}" 
        logging.info(f"         {string}") 

        # Get the indices of the valid (non-NaN) points
        valid_mask = ~np.isnan(z)
        coords_valid = np.array(np.nonzero(valid_mask)).T

        # Get the corresponding values for those points
        values_valid = z[valid_mask]

        # Get the indices of the invalid (NaN) points
        nan_mask = np.isnan(z)
        coords_nan = np.array(np.nonzero(nan_mask)).T

        # Interpolate the NaN values based on the surrounding valid points
        z[nan_mask] = griddata(coords_valid, values_valid, coords_nan, method='nearest')

        config.grid.area = z
        config.bag = np.where( b == 1, 1, 0)

        config.precipitation.area = np.ones( z.shape ) 
        config.absorption.area    = np.ones( z.shape ) 
        config.flow_measure.area  = np.ones( z.shape ) 
        
        config.grid.Nx = ds.x.size
        config.grid.Ny = ds.y.size

        config.grid.dx = (ds.x.values[1] - ds.x.values[0])
        config.grid.dy = (ds.y.values[0] - ds.y.values[1])

        config.absorption.enabled = True 
        config.precipitation.enabled = True

        #  config.result_dir = f"results\\area\\{config.run_type}"
        config.result_dir = os.path.join( config.result_dir, 
            "area", f"{config.run_type}", f"{config.date_time}" )
        
        os.makedirs(os.path.dirname(config.result_dir ), exist_ok=True)

        save_2D_plot(config, "_Ground_Level", config.grid.area )
        save_2D_plot(config, "_Obstacles_BAG", 1 - config.bag )

    else:
        string = "Can not handle run.type (see the config file)"
        raise ValueError(f"\n\nError: {string}\n")

    logging.info(f"") 
    logging.info(f"Info:    {config.run_type} config.result_dir:")
    logging.info(f"         {config.result_dir}")

    return config

def init_eta(Nx: int, Ny: int, A: float, sigma_i: float, sigma_j: float) -> np.ndarray:
    """
    Initialize the surface elevation eta at cell centers (i, j).

    Return
        e:  array[ : Ny, : Nx ]     Initional Eta values
    """
    e = np.zeros((Ny, Nx))  # Shape: (Ny, Nx)
    i_center = Nx // 2
    j_center = Ny // 2

    for j in range(Ny):
        for i in range(Nx):
            e[j, i] = A * np.exp(-((i - i_center) ** 2 / (2 * sigma_i ** 2))
                                        - ((j - j_center) ** 2 / (2 * sigma_j ** 2)))
    return e


def bottom_topography(config: Config) -> np.ndarray :
    """
    Generate the bottom topography for the simulation.

    Parameters:
        config.run_type   (test, benchmarkX (where X = 1, 2, 3, 4, 7a, 7b, 7c)

    Returns:
        z: Bottom topography array (num_points_y, num_points_x)
    """

    # print(f"BT  config: {config}")
    logging.info(f"Info:    bottom_topography():") 
    logging.info(f"             config.run_type: {config.run_type}")

    if config.run_type == "test":
        z = np.ones((config.grid.Ny, config.grid.Nx))
        print(f"Plot OpenBook Testcase 0")

    elif config.run_type == "benchmark1":
        #  Slope 
        #  max_height = 40.0
        
        z = generate_bottom_testcase(config) 

        # Plot the result
        print(f"Plot Testcase 1")

    elif config.run_type == 'benchmark2':   #  TestCase2 "open_book" 

        # Generate the grid
        z = generate_bottom_testcase(config) 

        # Plot the result
        print(f"Plot OpenBook Testcase 2")
    
    elif config.run_type ==  'benchmark3 ':

        # Generate the grid
        z = generate_bottom_testcase(config)

        print(f"Plot OpenBook Testcase 3")

    return z 



