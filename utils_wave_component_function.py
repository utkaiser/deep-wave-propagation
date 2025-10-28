import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def WaveEnergyComponentField_tensor(u, ut, c, dx):
    """
    Parameters
    ----------
    u : (pytorch tensor) physical wave component, displacement of wave
    ut : (pytorch tensor) physical wave component derived by t, velocity of wave
    c : (pytorch tensor) velocity profile dependent on x_1 and x_2
    dx : (float) time step in both dimensions / grid spacing

    Returns
    -------
    batchwise transformation to energy components of wave given parameters c and dx (tensor)
    function to use when applying end-to-end model
    """

    wx = torch.zeros(u.shape)
    wy = torch.zeros(u.shape)
    wtc = torch.zeros(u.shape)
    for b in range(u.shape[0]):
        wx[b, :, :], wy[b, :, :] = torch.gradient(u[b, :, :], spacing=dx)
        wtc[b, :, :] = torch.divide(ut[b, :, :], c[b, :, :])
    return wx, wy, wtc


def WaveSol_from_EnergyComponent_tensor(wx, wy, wtc, c, dx, sumv):
    """
    Parameters
    ----------
    wx : (pytorch tensor) energy wave component, u derived by x_1
    wy : (pytorch tensor) energy wave component, u derived by x_2
    wtc : (pytorch tensor) energy wave component, u derived by t and divided by c
    c : (numpy tensor) velocity profile dependent on x_1 and x_2
    dx : (float) time step in both dimensions / grid spacing
    sumv : (float) sum of u0

    Returns
    -------
    compute wave solution components from energy component
    """

    def _grad2func_tensor(vx, vy, dx, sumv):
        """
        Parameters
        ----------
        vx : (numpy tensor) energy wave component, u derived by x_1
        vy : (numpy tensor) energy wave component,  u derived by x_2
        dx : (float) time step in both dimensions / grid spacing
        sumv : (float) sum of u0

        Returns
        -------
        mapping gradient to functional value (numpy)
        """

        hatx = torch.fft.fft2(vx)
        haty = torch.fft.fft2(vy)
        ny, nx = vx.shape

        xii = (
            2
            * torch.pi
            / (dx * nx)
            * torch.fft.fftshift(torch.linspace(-round(nx / 2), round(nx / 2 - 1), nx))
        )
        yii = (
            2
            * torch.pi
            / (dx * ny)
            * torch.fft.fftshift(torch.linspace(-round(ny / 2), round(ny / 2 - 1), ny))
        )
        yiyi, xixi = torch.meshgrid(xii, yii, indexing="xy")

        radsq = torch.multiply(xixi, xixi) + torch.multiply(yiyi, yiyi)
        radsq[0, 0] = 1
        hatv = -1j * torch.divide(
            (
                torch.multiply(hatx.to(device), xixi.to(device))
                + torch.multiply(haty.to(device), yiyi.to(device))
            ),
            radsq.to(device),
        )
        hatv[0, 0] = sumv

        return torch.real(torch.fft.ifft2(hatv))

    u = torch.zeros((wx.shape[0], wx.shape[-1], wx.shape[-1]))

    for b in range(wx.shape[0]):
        u[b, :, :] = _grad2func_tensor(wx[b, :, :], wy[b, :, :], dx, sumv)

    ut = torch.multiply(wtc, c)

    return u, ut


def WaveEnergyComponentField_end_to_end(u, ut, c, dx):
    """
    Parameters
    ----------
    u : (numpy tensor) physical wave component, displacement of wave
    ut : (numpy tensor) physical wave component derived by t, velocity of wave
    c : (numpy tensor) velocity profile dependent on x_1 and x_2
    dx : (float) time step in both dimensions / grid spacing

    Returns
    -------
    batchwise transformation to energy components of wave given parameters c and dx (numpy)
    function to use when applying end-to-end model
    """

    wx, wy = np.gradient(u, dx)
    wtc = np.divide(ut, c)

    return wx, wy, wtc


def WaveEnergyField(u, ut, c, dx):
    """

    Parameters
    ----------
    u : (numpy tensor) physical wave component, displacement of wave
    ut : (numpy tensor) physical wave component derived by t, velocity of wave
    c : (numpy tensor) velocity profile dependent on x_1 and x_2
    dx : (float) time step in both dimensions / grid spacing

    Returns
    -------
    energy-semi norm of wave given parameters c and dx (numpy)
    """

    ux, uy = np.gradient(u, dx)
    absux = np.abs(ux)
    absuy = np.abs(uy)
    absutc = np.divide(np.abs(ut), c)
    w = (
        np.multiply(absux, absux)
        + np.multiply(absuy, absuy)
        + np.multiply(absutc, absutc)
    )

    return w