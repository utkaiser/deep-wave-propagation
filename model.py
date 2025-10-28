from os import path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, save

from numerical_solvers import velocity_verlet_tensor
from utils_wave_component_function import WaveEnergyComponentField_tensor,WaveSol_from_EnergyComponent_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model(model, model_name, dir_path="results/"):
    """
    Parameters
    ----------
    model : end-to-end model instance
    model_name : name of end-to-end model
    dir_path : directory where to save model in

    Returns
    -------
    save {model} as ".pt"-file
    """

    model.to(torch.device("cpu"))
    saving_path = dir_path + "saved_model_" + model_name + ".pt"
    if not path.isfile(saving_path):
        return save(model.state_dict(), saving_path)
    else:
        raise MemoryError("File (.pt) already exists.")


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
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_datasets, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_datasets, batch_size=batch_size, shuffle=True
    )

    return train_loader, val_loader, test_loader


class Model_end_to_end(nn.Module):
    """
    main end-to-end module class that builds interaction of different components
    down sampling + coarse solver + up sampling
    """

    def __init__(self, param_dict, downsampling_type, upsampling_type):
        """
        Parameters
        ----------
        param_dict : (dict) contains parameters to set up model
        downsampling_type : (string) defines down sampling component
        upsampling_type : (string) defines up sampling component
        """

        super().__init__()

        self.downsampling_type = downsampling_type
        self.upsampling_type = upsampling_type
        self.param_dict = param_dict
        self.model_downsampling = Numerical_downsampling(self.param_dict["res_scaler"])
        self.model_downsampling.to(device)
        self.model_numerical = Numerical_solver(
            param_dict["boundary_c"],
            param_dict["c_delta_x"],
            param_dict["c_delta_t"],
            param_dict["delta_t_star"],
        )
        self.model_numerical.to(device)
        self.model_upsampling = UNet(wf=1, depth=3, scale_factor=self.param_dict["res_scaler"]).double()
        self.model_upsampling.to(device)

    def forward(self, x, u_elapse):
        """
        Parameters
        ----------
        x : (pytorch tensor) input x as defined in paper with three wave energy components and velocity profile

        Returns
        -------
        propagates waves one time step delta_t_star using end-to-end model
        """

        # restriction component
        if self.downsampling_type == "CNN":
            downsampling_res, skip = self.model_downsampling(
                x
            )
        else:
            downsampling_res, _ = self.model_downsampling(
                x
            )

        # velocity verlet
        prop_result = self.model_numerical(downsampling_res, u_elapse)

        # up sampling component
        if self.upsampling_type == "Interpolation":
            outputs = self.model_upsampling(prop_result)
        else:
            if self.downsampling_type == "CNN":
                outputs = self.model_upsampling(prop_result.to(device), skip_all=skip)
            else:
                outputs = self.model_upsampling(prop_result.to(device), skip_all=None)

        return outputs.to(device)


class Numerical_downsampling(torch.nn.Module):
    """
    class to down sample solution numerically using bilinear interpolation
    """

    def __init__(self, res_scaler):
        """
        Parameters
        ----------
        res_scaler : (int) scale factor by which input is down sampled (usually 2 or 4)
        """
        super(Numerical_downsampling, self).__init__()
        self.res_scaler = res_scaler

    def forward(self, x):
        """
        Parameters
        ----------
        x : (pytorch tensor) input to convolutional block

        Returns
        -------
        down samples solution using bilinear interpolation
        """

        u_x, u_y, u_t_c, vel = (
            x[:, 0, :, :],
            x[:, 1, :, :],
            x[:, 2, :, :],
            x[:, 3, :, :],
        )  # b x w x h
        new_res = x.shape[-1] // self.res_scaler
        restr_output = torch.zeros([u_x.shape[0], 4, new_res, new_res])
        restr_output[:, 0, :, :] = F.upsample(
            u_x[:, :, :].unsqueeze(dim=0), size=(new_res, new_res), mode="bilinear"
        )
        restr_output[:, 1, :, :] = F.upsample(
            u_y[:, :, :].unsqueeze(dim=0), size=(new_res, new_res), mode="bilinear"
        )
        restr_output[:, 2, :, :] = F.upsample(
            u_t_c[:, :, :].unsqueeze(dim=0), size=(new_res, new_res), mode="bilinear"
        )
        restr_output[:, 3, :, :] = F.upsample(
            vel[:, :, :].unsqueeze(dim=0), size=(new_res, new_res), mode="bilinear"
        )
        return restr_output, None


class Numerical_solver(torch.nn.Module):
    """
    wrapping of numerical solver (velocity Verlet method) as pytorch neural network type
    for easier training and allowing backpropagation of gradients while training
    """

    def __init__(self, boundary_condition, c_delta_x, c_delta_t, delta_t_star):
        """
        Parameters
        ----------
        boundary_condition : (string) choice of boundary condition, "periodic" or "absorbing"
        c_delta_x : (float) coarse spatial step size / grid spacing (in x_1 and x_2 dimension)
        c_delta_t : (float) temporal step size
        delta_t_star : (float) time step a solver propagates a wave and solvers are compared
        """

        super(Numerical_solver, self).__init__()
        self.boundary_condition = boundary_condition
        self.c_delta_x = c_delta_x
        self.c_delta_t = c_delta_t
        self.delta_t_star = delta_t_star

    def forward(self, restr_output, u_elapse):
        """
        Parameters
        ----------
        restr_output : (pytorch tensor) output of adjacent down sampling component

        Returns
        -------
        propagates the wave using the velocity verlet algorithm on a coarse grid representation
        """

        vel_c = torch.Tensor.double(restr_output[:, 3, :, :])  # b x w_c x h_c

        restr_fine_sol_u, restr_fine_sol_ut = WaveSol_from_EnergyComponent_tensor(
            torch.Tensor.double(restr_output[:, 0, :, :]).to(device),
            torch.Tensor.double(restr_output[:, 1, :, :]).to(device),
            torch.Tensor.double(restr_output[:, 2, :, :]).to(device),
            vel_c.to(device),
            self.c_delta_x,
            torch.sum(torch.sum(torch.Tensor.double(u_elapse))),
        )

        # G delta t (coarse iteration)
        ucx, utcx = velocity_verlet_tensor(
            restr_fine_sol_u.to(device),
            restr_fine_sol_ut.to(device),
            vel_c.to(device),
            self.c_delta_x,
            self.c_delta_t,
            self.delta_t_star,
            number=1,
            boundary_c=self.boundary_condition,
        )  # b x w_c x h_c, b x w_c x h_c

        # change to energy components
        wx, wy, wtc = WaveEnergyComponentField_tensor(
            ucx.to(device), utcx.to(device), vel_c.to(device), self.c_delta_x
        )  # b x w_c x h_c, b x w_c x h_c, b x w_c x h_c

        # create input for nn
        return torch.stack(
            (wx.to(device), wy.to(device), wtc.to(device), vel_c.to(device)), dim=1
        ).to(
            device
        )  # b x 4 x 64 x 64


class UNet(nn.Module):
    """
    JNet class
    adaptation of https://discuss.pytorch.org/t/unet-implementation/426;
    forked from https://github.com/jvanvugt/pytorch-unet
    """

    def __init__(
        self,
        in_channels=4,
        n_classes=3,
        depth=3,
        wf=0,
        acti_func="relu",
        scale_factor=2,
    ):
        """
        Parameters
        ----------
        in_channels : (int) number of channels in input
        n_classes : (int) number of channels in output
        depth : (int) number of levels
        wf : (int) channel multiplication factor each level
        acti_func : (string) activation function, usually "relu"
        scale_factor : (int) scale factor by which input is up sampled (usually 2 or 4)

        Returns
        -------
        return UNet upsampling component
        """

        super(UNet, self).__init__()
        self.depth = depth
        prev_channels = in_channels
        self.acti_func = acti_func
        self.down_path = nn.ModuleList()
        for i in range(depth):
            if i != 0:
                self.down_path.append(
                    nn.Conv2d(
                        prev_channels,
                        in_channels * 2 ** (wf + i),
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    )
                )
                prev_channels = in_channels * 2 ** (wf + i)
            self.down_path.append(
                UNetConvBlock(
                    prev_channels, in_channels * 2 ** (wf + i), self.acti_func
                )
            )
            prev_channels = in_channels * 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, in_channels * 2 ** (wf + i), self.acti_func)
            )
            prev_channels = in_channels * 2 ** (wf + i)

        self.last = nn.ModuleList()
        for i in range(int(scale_factor / 2)):
            self.last.append(nn.Upsample(mode="bilinear", scale_factor=2))
            self.last.append(
                UNetConvBlock(prev_channels, prev_channels, self.acti_func)
            )
        self.last.append(nn.Conv2d(prev_channels, n_classes, kernel_size=1, bias=False))

    def forward(self, x, skip_all=None):
        """
        Parameters
        ----------
        x : (pytorch tensor) input from numerical solver
        skip_all : (pytorch tensor) skip connection as seen in paper that skips numerical solver

        Returns
        -------
        up samples input from numerical solver and enhances solution
        """
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i % 2 == 0:
                blocks.append(x)
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 2])
        for i, layer in enumerate(self.last):
            if len(self.last) - 3 == i and torch.is_tensor(skip_all):
                x = skip_all + x
            x = layer(x)
        return x


class UNetConvBlock(nn.Module):
    """
    Convolution blocks
    """

    def __init__(self, in_size, out_size, acti_func="relu"):
        """
        Parameters
        ----------
        in_size : (int) channel size of input
        out_size : (int) channel size of output
        acti_func : (string) activation function
        """

        super(UNetConvBlock, self).__init__()
        block = []

        if acti_func == "identity":
            block.append(
                nn.Conv2d(in_size, out_size, kernel_size=3, bias=False, padding=1)
            )
            block.append(
                nn.Conv2d(out_size, out_size, kernel_size=3, bias=False, padding=1)
            )
        elif acti_func == "relu":
            block.append(
                nn.Conv2d(in_size, out_size, kernel_size=3, bias=True, padding=1)
            )
            block.append(nn.BatchNorm2d(out_size))
            block.append(nn.ReLU())
            block.append(
                nn.Conv2d(out_size, out_size, kernel_size=3, bias=True, padding=1)
            )
            block.append(nn.BatchNorm2d(out_size))
            block.append(nn.ReLU())
        else:
            print("Choose either identity or relu \n")

        self.block = nn.Sequential(*block)

    def forward(self, x):
        """
        Parameters
        ----------
        x : (pytorch tensor) input to convolutional block

        Returns
        -------
        propagates the solution inside one convolutional block
        """
        return self.block(x)


class UNetUpBlock(nn.Module):
    """
    Upstream branch of JNet
    """

    def __init__(self, in_size, out_size, acti_func="relu"):
        """
        Parameters
        ----------
        in_size : (int) channel size of input
        out_size : (int) channel size of output
        acti_func : (string) activation function
        """

        super(UNetUpBlock, self).__init__()
        self.up = nn.Sequential(nn.Upsample(mode="bilinear", scale_factor=2))
        self.conv_block = UNetConvBlock(in_size, out_size, acti_func)

    def forward(self, x, bridge):
        """
        Parameters
        ----------
        x : (pytorch tensor) input to convolutional block
        bridge : (pytorch tensor) skip connection from adjacent convolutional block

        Returns
        -------
        propagates the solution inside one convolutional block
        """

        up = self.up(x)
        out = self.conv_block(up)

        out = out + bridge
        return out
