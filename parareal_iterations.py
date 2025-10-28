import torch
from matplotlib import pyplot as plt

from generate_data import one_iteration_pseudo_spectral_tensor
from model import Model_end_to_end
from numerical_solvers import WaveEnergyField_tensor
from utils_wave_component_function import WaveSol_from_EnergyComponent_tensor


def get_model(
    param_dict,
    down_sampling_component="Interpolation",
    up_sampling_component="UNet3",
    model_path="results/saved_model_test.pt",
):
    """
    Parameters
    ----------
    param_dict : (dict) contains parameters to set up model
    model_res : (int) resolution model can handle
    down_sampling_component: (string) choice of down sampling component
    up_sampling_component: (string)  choice of up sampling component
    model_path : (string) path to model parameters saved in ".pt"-file

    Returns
    -------
    load pre-trained model and retunr model
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model_end_to_end(
        param_dict, down_sampling_component, up_sampling_component,
    ).double()
    model = torch.nn.DataParallel(model).to(device)  # multi-GPU use
    model.load_state_dict(torch.load(model_path))

    return model

def pseudo_spectral_solutions(u_energy, u_elapse, vel, n_it, dx, dt, dt_star):
    print(f"Solve wave equation using pseudo-spectral method.")

    b, n_comp, w, h = u_energy.shape
    target = torch.zeros([n_it, n_comp, w, h])
    target_u_elapse = torch.zeros([n_it, w, h])

    for i in range(n_it):

        u_energy, u_elapse = one_iteration_pseudo_spectral_tensor(
            u_n_k=torch.cat(
                [u_energy, torch.from_numpy(vel).unsqueeze(dim=0).unsqueeze(dim=0)],
                dim=1,
            ),
            u_elapse=u_elapse,
            f_delta_x=dx,
            f_delta_t=dt,
            delta_t_star=dt_star,
        )
        target[i] = u_energy.clone()
        target_u_elapse[i] = u_elapse.clone()

    return target, target_u_elapse

def visualize_parareal(
    pseudo_spectral_tensor, parareal_tensor, n_parareal, n_it, f_delta_x, vel, u
):
    fig = plt.figure(figsize=(30, 10))

    for s in range(n_it-1):
        ax = fig.add_subplot(n_parareal + 1, n_it-1, 1 + s)
        ax.set_title(f"PS propagation {s+1}")

        u_energy = pseudo_spectral_tensor[s]
        u, ut = WaveSol_from_EnergyComponent_tensor(
            u_energy[0].unsqueeze(dim=0),
            u_energy[1].unsqueeze(dim=0),
            u_energy[2].unsqueeze(dim=0),
            torch.from_numpy(vel).unsqueeze(dim=0),
            f_delta_x,
            torch.sum(u),
        )
        w = WaveEnergyField_tensor(
            u.squeeze(), ut.squeeze(), torch.from_numpy(vel), f_delta_x
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(w)

    for s in range(1,n_it):
        for k in range(n_parareal):
            if s >= k+1 or k == 0:
                ax = fig.add_subplot(n_parareal + 1, n_it-1, 9 + 9 * k + s)
                ax.set_title(f"Parareal it {k}, prop. {s}")

                u_energy = parareal_tensor[k, s]
                u, ut = WaveSol_from_EnergyComponent_tensor(
                    u_energy[0, 0].unsqueeze(dim=0),
                    u_energy[0, 1].unsqueeze(dim=0),
                    u_energy[0, 2].unsqueeze(dim=0),
                    torch.from_numpy(vel).unsqueeze(dim=0),
                    f_delta_x,
                    torch.sum(torch.sum(u)),
                )
                w = WaveEnergyField_tensor(
                    u.squeeze(), ut.squeeze(), torch.from_numpy(vel), f_delta_x
                )
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(w)

    plt.show()
