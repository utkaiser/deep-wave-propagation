import numpy as np
import scipy.ndimage
import torch
from skimage.transform import resize

from numerical_solvers import init_pulse_gaussian, pseudo_spectral_tensor
from utils_wave_component_function import WaveEnergyComponentField_tensor, WaveSol_from_EnergyComponent_tensor


def generate_velocity_profile_crop(v_images, m, output_path, num_times):
    """
    Parameters
    ----------
    v_images : (tensor) full-size velocity profile that needs to be cropped
    m : (int) resolution, usually 128 (*1, *2 or *3)
    output_path : (string) path where to save generated crops
    num_times : (int) number of crops

    Returns
    -------
    sample velocity profiles by cropping randomly rotated and scaled images
    """

    wavespeed_list = []

    for img in v_images:
        for j in range(num_times):
            scale = (
                0.08 + 0.04 * np.random.rand()
            )  # chose this scaling because performed well
            angle = np.random.randint(4) * 22.5  # in degrees
            M = int(m / scale)  # how much we crop before resizing to m
            npimg = scipy.ndimage.rotate(
                img, angle, cval=1.0, order=4, mode="wrap"
            )  # bilinear interp and rotation
            h, w = npimg.shape

            # crop but make sure it is not blank image
            while True:
                xTopLeft = np.random.randint(max(1, w - M))
                yTopLeft = np.random.randint(max(1, h - M))
                newim = npimg[yTopLeft : yTopLeft + M, xTopLeft : xTopLeft + M]

                if (
                    newim.std() > 0.005
                    and newim.mean() < 3.8
                    and not np.all(newim == 0)
                ):
                    npimg = 1.0 * newim
                    break

            wavespeed_list.append(resize(npimg, (m, m), order=4))

    np.savez(output_path, wavespeedlist=wavespeed_list)


def crop_center(img, crop_size, scaler=2):
    """
    Parameters
    ----------
    img : (numpy / pytorch tensor) input image to crop
    crop_size : (int) size of crop
    scaler : scale factor

    Returns
    -------
    crop center of img given size of crop, and scale factor
    """

    y, x = img.shape
    startx = x // scaler - (crop_size // scaler)
    starty = y // scaler - (crop_size // scaler)

    return img[starty : starty + crop_size, startx : startx + crop_size]


def initial_condition_gaussian(vel, mode, res_padded):
    """
    Parameters
    ----------
    vel : (numpy tensor) velocity profile
    resolution : (int) resolution of actual area to propagate wave
    optimization : (string) optimization technique; "parareal" or "none"
    mode : (string) defines initial condition representation; "physical_components" or "energy_components"
    res_padded : (int) resolution of padded area to propagate wave, we need a larger resolution in case of "parareal" and / or "absorbing"

    Returns
    -------
    generates a Gaussian pulse to be used as an initial condition for our end-to-end model to advance waves
    """

    dx, width, center_x, center_y = 2.0 / 128.0, 7000, 0, 0
    u0, ut0 = init_pulse_gaussian(width, res_padded, center_x, center_y)

    if mode == "physical_components":
        return u0, ut0
    else:  # energy_components
        u0, ut0 = torch.from_numpy(u0).unsqueeze(dim=0), torch.from_numpy(
            ut0
        ).unsqueeze(dim=0)
        wx, wy, wtc = WaveEnergyComponentField_tensor(
            u0, ut0, vel.unsqueeze(dim=0), dx=dx
        )
        return torch.stack([wx, wy, wtc], dim=1), u0


def one_iteration_pseudo_spectral_tensor(
    u_n_k, u_elapse, f_delta_x=2.0 / 128.0, f_delta_t=(2.0 / 128.0) / 20.0, delta_t_star=0.06
):
    """

    Parameters
    ----------
    u_n_k : (pytorch tensor) wave representation as energy components
    f_delta_x : (float) spatial step size / grid spacing (in x_1 and x_2 dimension)
    f_delta_t : (float) temporal step size
    delta_t_star : (float) time step a solver propagates a wave and solvers are compared

    Returns
    -------
    propagates a wave for one time step delta_t_star using the pseudo-spectral method
    """

    u, u_t = WaveSol_from_EnergyComponent_tensor(
        u_n_k[:, 0, :, :].clone(),
        u_n_k[:, 1, :, :].clone(),
        u_n_k[:, 2, :, :].clone(),
        u_n_k[:, 3, :, :].clone(),
        f_delta_x,
        torch.sum(torch.sum(torch.sum(u_elapse))),
    )
    vel = u_n_k[:, 3, :, :].clone()
    u_prop, u_t_prop = pseudo_spectral_tensor(
        u, u_t, vel, f_delta_x, f_delta_t, delta_t_star
    )
    u_x, u_y, u_t_c = WaveEnergyComponentField_tensor(
        u_prop, u_t_prop, vel.unsqueeze(dim=1), f_delta_x
    )
    return torch.stack([u_x, u_y, u_t_c], dim=1), u_prop
