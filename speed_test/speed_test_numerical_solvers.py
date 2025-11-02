import torch
import argparse
import time
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_data import get_velocity_model, init_pulse_gaussian
from numerical_solvers_old import pseudo_spectral_tensor as pseudo_spectral_tensor_old
from numerical_solvers_old import velocity_verlet_tensor as velocity_verlet_tensor_old
from numerical_solvers import pseudo_spectral_tensor, velocity_verlet_tensor

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def test_speed_of_numerical_solver(
    vel_data_path = "data/crop_test.npz",
    method = "pseudo-spectral",
    dx = 2./128.,
    dt = 1/600.,
    dt_star = .06,
    boudary_c = "periodic",
    n_trials = 10,
    num_decimals_round = 6,
    tolerance = 1e-4,
):
    '''
    Parameters
    ----------
    vel_data_path : (string) path to velocity profile crops
    method : (string) "pseudo-spectral" or "velocity-verlet"
    res : (int) dimensionality of the input
    dx : (float) spatial step size numerical solver
    dt : (float) temporal step size numerical solver
    dt_star : (float) time interval the solver is applied once
    boundary_c : (string) boundary conditions, "periodic" or "absorbing"

    Returns
    -------
    Speed test of numerical solver for advancing wave fields
    '''

    # computing initial condition using gaussian pulse
    u_init, ut_init = init_pulse_gaussian(7000, 128, 0, 0)
    u_init, ut_init = torch.from_numpy(u_init), torch.from_numpy(ut_init)
    vel = torch.from_numpy(get_velocity_model(vel_data_path, visualize=False)).to(device)

    # set up lists to store results
    speed_results_old, speed_results_optimized, interations_all_close = [], [], []

    for trial in range(n_trials):
        speed_results_old.append([]), speed_results_optimized.append([]), interations_all_close.append([])
        u_old, ut_old = u_init.clone().to(device), ut_init.clone().to(device)
        u, ut = u_init.clone().to(device), ut_init.clone().to(device)

        for s in range(10):
            # ---- old implementation speed test ----
            # run one iteration of the RK4 / velocity Verlet method for time dt_star and step size dx, time increment dt
            start_time = time.perf_counter()
            if method == "pseudo-spectral":
                u_old, ut_old = pseudo_spectral_tensor_old(u_old, ut_old, vel, dx, dt, dt_star)
            else:  # method == "velocity_verlet"
                u_old, ut_old = velocity_verlet_tensor_old(u_old, ut_old, vel, dx, dt, dt_star, boundary_c=boudary_c)
            end_time = time.perf_counter()
            speed_results_old[trial].append(round(end_time - start_time, num_decimals_round))

            # ---- optimized implementation speed test ----
            # run one iteration of the RK4 / velocity Verlet method for time dt_star and step size dx, time increment dt
            start_time = time.perf_counter()
            if method == "pseudo-spectral":
                u, ut = pseudo_spectral_tensor(u, ut, vel, dx, dt, dt_star)
            else:  # method == "velocity_verlet"
                u, ut = velocity_verlet_tensor(u, ut, vel, dx, dt, dt_star, boundary_c=boudary_c)
            end_time = time.perf_counter()
            speed_results_optimized[trial].append(round(end_time - start_time, num_decimals_round))
            interations_all_close[trial].append(torch.allclose(u, u_old, atol=tolerance, rtol=0.) and torch.allclose(ut, ut_old, atol=tolerance, rtol=0.))


    print(f"Old implementation: {speed_results_old}.\n")
    print(f"Optimized implementation: {speed_results_optimized}.\n")
    print(f"All iterations close: {interations_all_close}.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run speed test experiments with specified parameters.")
    parser.add_argument('method', type=str, help="Numerical method: pseudo-spectral or velocity-verlet")
    parser.add_argument('boundary_c', type=str, help="Boundary condition type: periodic or absorbing")
    args = parser.parse_args()
    method, boundary_c = args.method, args.boundary_c

    print(f"Running experiment with {method=}, {boundary_c=}")
    test_speed_of_numerical_solver(
        method=method,
        boudary_c=boundary_c
    )
