# set/simulation.py
import logging
import numpy as np
from math import log, ceil 
from .compute import shallow_water_displacement, get_total_flow_1D, get_total_flow_2D
from .compute import matrix_F, matrix_G, matrix_T
from .compute import vector_b, discrete_H, update_velocity, update_H
from .compute import matrix_T_vectorized, matrix_F_vectorized
from .compute import vector_b_vectorized, update_velocity_vectorized 
from .compute import saint_venant_update_vectorized_bv

from .solver import newton_solver, newton_solver_modified_trap

from .tools.init import set_simulation_type, init_eta
from .tools.check import exec_time, boundary_velocity, water_lvl_stable
from .tools.check import flow_stable, water_lvl_treshold, flow_treshold
from .tools.save import save_inundation_2D_plot, save_flow_velocity_2D_plot
from .tools.save import save_flow_1D_plot, save_sim_results

from .config import Config

def initialize_simulation(config: Config) : 

    config = set_simulation_type(config)

    z = np.copy( config.grid.area )
    e = np.zeros(config.grid.area.shape)
    h = np.copy( z ) + np.copy( e )

    Ny, Nx = h.shape 
    config.grid.Ny = Ny 
    config.grid.Nx = Nx 
    
    if config.run_type == 'benchmark0':
        # eta = init_eta(Nx, Ny, A=1, sigma_i=0.5, sigma_j=0.5)   #  A = 1 ->  A = 1 meter
        e[ int(Ny/2), int(Nx/2) ] = 1

    # u = np.zeros((Ny, Nx))
    # v = np.zeros((Ny, Nx))
    return config, z, e, h


def run(config: Config):
    """
    Run simulation loop for the 2D shallow water model.
    """
    #  Print ndarray values up to <precision> decimals
    np.set_printoptions(precision=6,suppress=True) 

    # Initialize dictionary to store method timing
    timing_results = {}

    elapsed_time, (config, z, e, h) = exec_time(initialize_simulation, config)
    # logging.debug
    timing_results["   init_sim"] = np.round(elapsed_time, 3)

    #  Define some shortcuts 
    tt = config.time.total 
    mp = config.max_plots

    dt = config.time.dt     #  time between iterations 
    dx = config.grid.dx     #  inter-cell spacing in X direction 
    dy = config.grid.dy     #  inter-cell spacing in Y direction  
    Sx = config.grid.Sx     #  sample-area mid-point in X direction 
    Sy = config.grid.Sy     #  sample-area mid-point in Y direction 
    Sd = config.grid.Sd     #  sample-area mid-point offeset 
    Bg = config.bag

    #  descrete areas (value = 0 or 1)  
    P = np.copy(config.precipitation.area) 
    A = np.copy(config.absorption.area) 
    M = np.copy(config.flow_measure.area) 
    B = np.copy(config.bag) 

    logging.debug(f"Shapes m, p, a: {M.shape}, {P.shape}, {A.shape}, {B.shape}")
    logging.debug(f"Shapes z, e, h: {z.shape}, {e.shape}, {h.shape}\n")
                 
    #  save_inundation_plot(config, e, dt, time, pc, maxscale_e)
    precipitation_area = np.zeros(P.shape)
    absorption_area = np.zeros(A.shape)
    #  flow_measurement_area = np.copy( F ) 
    #  logging.debug(f"Hc_updated {iter} :\n{np.round(Hc_updated, 2)}\n\n")

    pr = config.precipitation.rate / 3600000  # Convert mm/hr to m/s
    ar = config.absorption.rate / 3600000 # Convert mm/hr to m/s

    if config.precipitation.enabled == True : 
        string = f"Precipitation start: {config.precipitation.t_start} [s]"
        string = f"{string}, with {config.precipitation.rate} mm/hr @ grid cell" 
        logging.info(f"") 
        logging.info(f"Info:    {string}")
        string = f"Precipitation stop : {config.precipitation.t_end} [s]" 
        string = f"{string}, with 0 mm/hr @ grid cell" 
        logging.info(f"         {string}")
                     
    if config.absorption.enabled == True :
        string = f"Absorption Rate = {config.absorption.rate:.6f} mm/hr @ grid cell"
        logging.info(f"") 
        logging.info(f"Info:    {string}")

        absorption_area = A * ar * dt 

    inv_A = 1 - A   #  0 where full absorption is required 
                    #  (drain away, so inundation = 0), 1 where not.

    inv_B = 1 - B   #  0 where full absorption is required 
                    #  (obstacle, so inundation = 0), 1 where not.

    # logging.debug(f"Sy, Sx = {Sy}, {Sx}") 
    # Hc, H_u, H_v = discrete_H(h, eta)
    
    # logging.debug(f"h:\n{h[Sy-Sd:Sy+Sd,Sx-Sd:Sx+Sd]}\n")
    # logging.debug(f"Shapes Hc, H_u, H_v:\n{Hc.shape}, {H_u.shape}, {H_v.shape}\n")
    # logging.debug(f"Hc:\n{Hc[Sy-Sd:Sy+Sd,Sx-Sd:Sx+Sd]}\n")
    # logging.debug(f"eta:\n{eta[Sy-Sd:Sy+Sd,Sx-Sd:Sx+Sd]}\n")
    # logging.debug(f"p:\n{P[Sy-Sd:Sy+Sd,Sx-Sd:Sx+Sd]}\n")
    # logging.debug(f"f:\n{F[Sy-Sd:Sy+Sd,Sx-Sd:Sx+Sd]}\n")
    time = 0.0
    total_flow = 0.0
    netto_flow = 0.0
    pc = 0
    i = 0 

    ti = ceil(tt / dt)  #   total iterations (ti) = (tt + dt - 1) // dt    
    lti = log(ti) - 2   #   log of total iterations log(ti) minus 2 

    #   now a fair calculation is made, as a balance in number of plots 
    #   and itterations per plot 
    if min(lti, 0) == 0 :       #  number of iterations < 100 
        ipps = ceil(ti / mp)    #  iterations per plot = total iterations / max plots 
    else : 
        ipps = ceil(ti / mp / abs(min(lti, 0)))    
    
    tps = ceil(ti / ipps)       #  total plots 

    tis = tps * ipps    #  total nr of iterations = 
                        #  total plots * iterations per plot 

    logging.info(f"")
    logging.info(f"Info:            Iteration delta t: {dt:.3f} [s] ")
    logging.info(f"         ===================================== " )
    logging.info(f"              Iterations per Plot: {ipps:6d} [#] ") 
    logging.info(f"                  Number of Plots: { tps:6d} [#] ")
    logging.info(f"         ===================================== *")
    logging.info(f"         Total nmbr of Iterations: { tis:6d} [#]" ) 

    time_list = []
    flow_rate_eMuv = []
    flow_rate_eMxy = []

    time_list.append(time)
    flow_rate_eMuv.append(total_flow)
    flow_rate_eMxy.append(netto_flow)

    maxscale_e = 0 
    # maxscale_H = np.max( eta )
    # save_inundation_plot(config, Hc, h, dt, time, pc, maxscale_H)

    # past :  np.zeros((config.grid.Ny, config.grid.Nx))

    # logging.debug(f"    Iter {i},  z[sample_area]:\n{z[Sy-Sd:Sy+Sd,Sx-Sd:Sx+Sd]}\n")
    # logging.debug(f"    Iter {i},  e[sample_area]:\n{e[Sy-Sd:Sy+Sd,Sx-Sd:Sx+Sd]}\n")
    # logging.debug(f"    Iter {i},  h[sample_area]:\n{h[Sy-Sd:Sy+Sd,Sx-Sd:Sx+Sd]}\n")

    # Track the method timings for this iteration
    #  iter_timing = {}
    vf = 0 
    iter_progress_update = int( ipps / 10 ) 
    rain = "OFF"

    config.save_to_file()

    logging.info(f"")
    logging.info(f"Info:    Now Starting simulation !!! ")
    logging.info(f"")

    for i in range(1, tis + 1, 1):

        time += dt

        if config.precipitation.enabled and time >= config.precipitation.t_start and time < config.precipitation.t_end:
            # Add precipitation at every itteration
            #  print(f"    iter {iter} , precip_interval_iters {precip_interval_iters} ")
            precipitation_area = P * pr * dt
            #  absorption_area = A * ar * dt * 200/124 
            #     logging.debug(f"Added {precip_rate}mm precipitation at t = {time:.2f} s")
            rain = "ON"
        else :
            rain = "OFF"
            precipitation_area = np.zeros( P.shape )
            #  absorption_area = np.zeros( A.shape )

        e += precipitation_area
        #  e *= inv_A  #  No inundation if there is full absorption 
        e *= inv_B  #  No inundation if there is an obstacle 

        elapsed_time, (z, e, h, u, v) = exec_time(  
            shallow_water_displacement, config, z, e, h) 
        
        logging.debug(f"Debug:   i={i}, u[sample_area]: \n{u[Sy-Sd:Sy+Sd,Sx-Sd:Sx+Sd]}\n")
        logging.debug(f"Debug:   i={i}, v[sample_area]: \n{v[Sy-Sd:Sy+Sd,Sx-Sd:Sx+Sd]}\n")
        logging.debug(f"Debug:   i={i}, m[sample_area]: \n{M[Sy-Sd:Sy+Sd,Sx-Sd:Sx+Sd]}\n")

        db_eMuv = get_total_flow_1D(config, e, M, u, v)   #  flux in measurement grid
        db_eMxy = np.max( e * M ) * dx * dy  / dt     #  max flux in area grid [m3/s] 

        #  sum_ep = np.sum( e * P ) 
        # print(f"i={i} t={time:.2f}, db_eMuv={db_eMuv:.3f}, db_eMxy ={db_eMxy:.3f}, sum_e={sum_ep:.6f},  rain={rain}")

        time_list.append(time) 
        flow_rate_eMuv.append(db_eMuv)       #   accumulate vector flow (vf)
        flow_rate_eMxy.append(db_eMxy)

        #  Remove all inundation (eta) in the absorption area 
        #  e *= inv_A

        # # Compute the explicit terms for F
        # elapsed_time, (Fu, Fv) = exec_time(  matrix_F_vectorized, config, u, v  )
        # iter_timing["  matrix_F2"] = np.round(elapsed_time, 3)

        # # Compute flux terms G
        # elapsed_time, (Gu, Gv) = exec_time(  matrix_G, config, H_u, H_v, Fu, Fv)
        # iter_timing["  matrix_G2"] = np.round(elapsed_time, 3)

        # # Assemble matrix T 
        # elapsed_time, (T) = exec_time(  matrix_T_vectorized, config, Hc)
        # iter_timing["  matrix_T2"] = np.round(elapsed_time, 3)

        # # Compute vector b
        # elapsed_time, (b) = exec_time(  vector_b_vectorized, config, Hc, H_u, H_v, Gu, Gv)
        # iter_timing["  vector_b2"] = np.round(elapsed_time, 3)

        # # Solve for new surface elevation 
        # elapsed_time, eta_new = exec_time(  newton_solver, config, Hc, eta, h, b, T)
        # iter_timing["newton_solv"] = np.round(elapsed_time, 3)
        # logging.debug(f"eta_new   {iter}:\n{eta_new[Sy-Sd:Sy+Sd,Sx-Sd:Sx+Sd]}\n")

        # Update total water depths
        # elapsed_time, (Hc_new, H_u_new, H_v_new) = exec_time(  discrete_H, h, eta_new)
        # iter_timing[" discrete_H"] = np.round(elapsed_time, 3)
        # logging.debug(f"Hc_new  {iter} :\n{Hc_new[Sy-Sd:Sy+Sd,Sx-Sd:Sx+Sd]}\n")

        # # Update velocities 
        # elapsed_time, (u_new, v_new) = exec_time(update_velocity_vectorized, config, Hc_new, H_u_new, H_v_new, h, u, v, eta_new, Fu, Fv)
        # iter_timing[" velocity_2"] = np.round(elapsed_time, 3)

        # elapsed_time, (u_new, v_new) = exec_time(calc_velocity, u_new, v_new)
        # iter_timing["calc_velocity"] = np.round(elapsed_time, 3)

        # Apply boundary conditions
        # elapsed_time, (u_new, v_new) = exec_time(boundary_velocity, u_new, v_new)
        # iter_timing["bd_velocity"] = np.round(elapsed_time, 3)
        
        # Update water depth based on new velocities
        # elapsed_time, Hc_updated = exec_time(update_H , config, Hc_new, H_u_new, H_v_new, h, u_new, v_new)
        # iter_timing["   update_H"] = np.round(elapsed_time, 3)
        # logging.debug(f"Hc_updated {iter} :\n{Hc_updated[Sy-Sd:Sy+Sd,Sx-Sd:Sx+Sd]}\n")

        # Compute total flow rate
        # logging.debug(f"flow_measurement_area {iter} :\n{flow_measurement_area[Sy-Sd:Sy+Sd,Sx-Sd:Sx+Sd]}\n"

        # logging.debug(f"    Iter {i},  e[sample_area]:\n{e[Sy-Sd:Sy+Sd,Sx-Sd:Sx+Sd]}\n")
        #  logging.debug(f"    Iter {i},  e[entire_area]:\n{e}\n")

        # logging.debug(f"u[sample] i={i} :\n{u[Sy-Sd:Sy+Sd,Sx-Sd:Sx+Sd]}\n")
        #  print(f"vf {vf:.2f}, nf {nf:.2f}")

        #  calculate the sum of all inundation levels together
        #  total_sum_h = np.sum(Hc_updated - h) 
        
        if ( i % iter_progress_update == 0 ): 
            
            pct = 100 * i/tis
            infstr = f"i={i}/{tis} ({pct:.2f}%), t={time:.1f} [s], db_eMuv={db_eMuv:.3f}, db_eMxy={db_eMxy:.3f}, rain={rain}"
            logging.info(f"Info: {infstr}")

        # still ToDo:  Check for stopping conditions
        # if water_lvl_treshold(eta, eta_new, threshold=0.0001) and \
        #       flow_treshold(u, u_new, v, v_new, threshold=0.0001):
        #     logging.info("\n\nStopping condition met: Water level and flow less than threshold!\n\n")
        #     break

        if i > 0 and ( i % ipps ) == 0:
            maxscale_e = np.max( e )
            f = get_total_flow_2D(config, e, M, u, v)
            maxscale_f = np.max( f )
            pct = 100 * i/tis
            pc += 1
            #  pcs = f"{pc:03d}"
            save_inundation_2D_plot(config,    e, dt, time, pc, maxscale_e )
            save_flow_velocity_2D_plot(config, f, dt, time, pc, maxscale_f ) 
            save_flow_1D_plot(config, f"vF", pc, time_list, flow_rate_eMuv)
            save_flow_1D_plot(config, f"eA", pc, time_list, flow_rate_eMxy)
            s = f"p={pc}/{mp} Saved Plots Inundation & Flow_Velocity t={time:.2f} [s]"
            logging.info(f"Info: {s}")

        # Update variables for the next iteration
        #  eta = eta_new.copy()
        # Hc = Hc_new.copy()
        # H_u = H_u_new.copy()
        # H_v = H_v_new.copy()
        # u = u_new.copy()
        # v = v_new.copy()

        # logging.info(f"Time: {np.round(time_list[-1],2)}, Flow rate: {np.round(flow_rate_list[-1], 2)}")
        # logging.debug(f"Net flow: {np.round(net_flow_list[-1], 2)}\n")

    #   flow_rate_eMuv.append(db_eMuv)       #   accumulate vector flow (vf)
    #   flow_rate_eMxy.append(db_eMxy)

    pc += 1
    save_inundation_2D_plot(config, e, dt, time, pc, maxscale_e)
    save_flow_velocity_2D_plot(config, f, dt, time, pc, maxscale_f ) 
    save_flow_1D_plot(config, f"vF", pc, time_list, flow_rate_eMuv)
    save_flow_1D_plot(config, f"eA", pc, time_list, flow_rate_eMxy)

    save_sim_results(config, timing_results)

    logging.info(f"\nInfo:          Simulation completed !\n")



