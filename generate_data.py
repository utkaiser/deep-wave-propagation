import numpy as np
import scipy.ndimage
import torch
from skimage.transform import resize
import matplotlib.pyplot as plt

from numerical_solvers import pseudo_spectral_tensor
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


def get_velocity_model(data_path, visualize=True):
    """
    Parameters
    ----------
    data_path : (string) path to velocity profile crops
    visualize : (boolean) whether to visualize data

    Returns
    -------
    (numpy array) single velocity profile
    """

    # choose first velocity profile out of list of velocity crops
    vel = np.load(data_path)["wavespeedlist"].squeeze()[0]

    if visualize:
        plt.axis("off")
        plt.title("Velocity profile")
        plt.imshow(vel)
        plt.show()

    return vel


def init_pulse_gaussian(width, res_padded, center_x, center_y):
    """

    Parameters
    ----------
    width : (float) width of initial pulse
    res_padded : (int) padded resolution
    center_x : (float) center of initial pulse in x_1 direction
    center_y : (float) center of initial pulse in x_2 direction

    Returns
    -------
    generates initial Gaussian pulse  (see formula in paper)
    """

    xx, yy = np.meshgrid(np.linspace(-1, 1, res_padded), np.linspace(-1, 1, res_padded))
    u0 = np.exp(-width * ((xx - center_x) ** 2 + (yy - center_y) ** 2))
    ut0 = np.zeros([np.size(xx, axis=1), np.size(yy, axis=0)])
    return u0, ut0


def fetch_data_end_to_end(data_paths, batch_size, additional_test_paths):
    """
    Parameters
    ----------
    data_paths : (string) data paths to use for training and validation
    batch_size : (int) batch size
    additional_test_paths : (string) data paths to use for testing

    Returns
    -------
    return torch.Dataloader object to iterate over training, validation and testing samples
    """

    def get_datasets(data_paths):
        # concatenate paths
        datasets = []
        for i, path in enumerate(data_paths):
            np_array = np.load(path)  # 200 x 11 x 128 x 128
            datasets.append(
                torch.utils.data.TensorDataset(
                    torch.stack(
                        (
                            torch.from_numpy(np_array["Ux"]),
                            torch.from_numpy(np_array["Uy"]),
                            torch.from_numpy(np_array["Utc"]),
                            torch.from_numpy(np_array["vel"]),
                            torch.from_numpy(np_array["u_phys"])
                        ),
                        dim=2,
                    )
                )
            )
        return torch.utils.data.ConcatDataset(datasets)

    # get full dataset
    full_dataset = get_datasets(data_paths)

    # get split sizes
    train_size = int(0.8 * len(full_dataset))
    val_or_test_size = int(0.1 * len(full_dataset))

    # split dataset randomly and append special validation/ test data
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_or_test_size, val_or_test_size]
    )
    val_datasets = val_dataset  # + get_datasets(additional_test_paths)
    test_datasets = test_dataset + get_datasets(additional_test_paths)

    # get dataloader objects
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_datasets, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_datasets, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True
    )

    return train_loader, val_loader, test_loader