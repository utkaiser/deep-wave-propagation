import matplotlib.pyplot as plt
import numpy as np
import torch


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


def velocity_verlet_tensor(
    u0, ut0, vel, dx, dt, delta_t_star, number=0, boundary_c="periodic"
):
    """
    Parameters
    ----------
    u0 : (pytorch tensor) physical wave component, displacement of wave
    ut0 : (pytorch tensor) physical wave component derived by t, velocity of wave
    vel : (pytorch tensor) velocity profile dependent on x_1 and x_2
    dx : (float) time step in both dimensions / grid spacing
    dt : (float) temporal step size
    delta_t_star : (float) time step a solver propagates a wave and solvers are compared
    number : (int) change number from 0 to 1 if batch added as a dimensionality
    boundary_c : (string) choice of boundary condition, "periodic" or "absorbing"

    Returns
    -------
    propagate wavefield using velocity Verlet in time and the second order discrete Laplacian in space
    """

    def _periLaplacian_tensor(v, dx, number):
        """
        Parameters
        ----------
        v : (pytorch tensor) velocity profile dependent on x_1 and x_2
        dx : (float) spatial step size / grid spacing (in x_1 and x_2 dimension)
        number : (int) change number from 0 to 1 if batch added as a dimensionality

        Returns
        -------
        compute periodic Laplacian evaluate discrete Laplacian with periodic boundary condition
        """

        Lv = (
            torch.roll(v, 1, dims=1 + number)
            - 2 * v
            + torch.roll(v, -1, dims=1 + number)
        ) / (dx**2) + (
            torch.roll(v, 1, dims=0 + number)
            - 2 * v
            + torch.roll(v, -1, dims=0 + number)
        ) / (
            dx**2
        )

        return Lv

    Nt = round(abs(delta_t_star / dt))
    c2 = torch.mul(vel, vel)
    u, ut = u0, ut0

    if boundary_c == "periodic":
        for i in range(Nt):
            # Velocity Verlet

            ddxou = _periLaplacian_tensor(u, dx, number)
            u = u + dt * ut + 0.5 * dt**2 * torch.mul(c2, ddxou)
            ddxu = _periLaplacian_tensor(u, dx, number)
            ut = ut + 0.5 * dt * torch.mul(c2, ddxou + ddxu)

        return u, ut

    elif boundary_c == "absorbing":
        # shape: u, ut -> b x w_c x h_c

        Ny, Nx = u0.shape[-1] - 1, u0.shape[-2] - 1

        lambda_v = abs(dt / dx)
        lambda2 = lambda_v**2
        lambdaC2 = lambda2 * c2

        a = dx / (dx + abs(dt))

        # Euler step to generate u1 from u0 and ut0
        uneg1 = u0 - dt * ut0
        u2 = u0.clone()
        u1 = u0.clone()
        u0 = uneg1.clone()

        for k in range(Nt):
            # wave equation update
            u2[:, 1:Ny, 1:Nx] = (
                2 * u1[:, 1:Ny, 1:Nx]
                - u0[:, 1:Ny, 1:Nx]
                + lambdaC2[:, 1:Ny, 1:Nx]
                * (
                    u1[:, 2 : Ny + 1, 1:Nx]
                    + u1[:, 0 : Ny - 1, 1:Nx]
                    + u1[:, 1:Ny, 2 : Nx + 1]
                    + u1[:, 1:Ny, 0 : Nx - 1]
                    - 4 * u1[:, 1:Ny, 1:Nx]
                )
            )

            # absorbing boundary update (Engquist-Majda ABC second order)
            Ny, Nx = Ny - 1, Nx - 1
            u2[:, -1, 1 : Nx + 1] = a * (
                -u2[:, Ny, 1 : Nx + 1]
                + 2 * u1[:, -1, 1 : Nx + 1]
                - u0[:, -1, 1 : Nx + 1]
                + 2 * u1[:, Ny, 1 : Nx + 1]
                - u0[:, Ny, 1 : Nx + 1]
                + lambda_v
                * (
                    u2[:, Ny, 1 : Nx + 1]
                    - u0[:, Ny, 1 : Nx + 1]
                    + u0[:, -1, 1 : Nx + 1]
                )
                + 0.5
                * lambda2
                * (
                    u0[:, -1, 2 : Nx + 2]
                    - 2 * u0[:, -1, 1 : Nx + 1]
                    + u0[:, -1, 0:Nx]
                    + u2[:, Ny, 2 : Nx + 2]
                    - 2 * u2[:, Ny, 1 : Nx + 1]
                    + u2[:, Ny, 0:Nx]
                )
            )

            u2[:, 0, 1 : Nx + 1] = a * (
                -u2[:, 1, 1 : Nx + 1]
                + 2 * u1[:, 0, 1 : Nx + 1]
                - u0[:, 0, 1 : Nx + 1]
                + 2 * u1[:, 1, 1 : Nx + 1]
                - u0[:, 1, 1 : Nx + 1]
                + lambda_v
                * (u2[:, 1, 1 : Nx + 1] - u0[:, 1, 1 : Nx + 1] + u0[:, 0, 1 : Nx + 1])
                + 0.5
                * lambda2
                * (
                    u0[:, 0, 2 : Nx + 2]
                    - 2 * u0[:, 0, 1 : Nx + 1]
                    + u0[:, 0, 0:Nx]
                    + u2[:, 1, 2 : Nx + 2]
                    - 2 * u2[:, 1, 1 : Nx + 1]
                    + u2[:, 1, 0:Nx]
                )
            )

            u2[:, 1 : Ny + 1, -1] = a * (
                -u2[:, 1 : Ny + 1, Nx]
                + 2 * u1[:, 1 : Ny + 1, Nx]
                - u0[:, 1 : Ny + 1, Nx]
                + 2 * u1[:, 1 : Ny + 1, Nx + 1]
                - u0[:, 1 : Ny + 1, Nx + 1]
                + lambda_v
                * (
                    u2[:, 1 : Ny + 1, Nx]
                    - u0[:, 1 : Ny + 1, Nx]
                    + u0[:, 1 : Ny + 1, Nx + 1]
                )
                + 0.5
                * lambda2
                * (
                    u0[:, 2 : Ny + 2, Nx + 1]
                    - 2 * u0[:, 1 : Ny + 1, Nx + 1]
                    + u0[:, 0:Ny, Nx + 1]
                    + u2[:, 2 : Ny + 2, Nx]
                    - 2 * u2[:, 1 : Ny + 1, Nx]
                    + u2[:, 0:Ny, Nx]
                )
            )

            u2[:, 1 : Ny + 1, 0] = a * (
                -u2[:, 1 : Ny + 1, 1]
                + 2 * u1[:, 1 : Ny + 1, 1]
                - u0[:, 1 : Ny + 1, 1]
                + 2 * u1[:, 1 : Ny + 1, 0]
                - u0[:, 1 : Ny + 1, 0]
                + lambda_v
                * (u2[:, 1 : Ny + 1, 1] - u0[:, 1 : Ny + 1, 1] + u0[:, 1 : Ny + 1, 0])
                + 0.5
                * lambda2
                * (
                    u0[:, 2 : Ny + 2, 0]
                    - 2 * u0[:, 1 : Ny + 1, 0]
                    + u0[:, 0:Ny, 0]
                    + u2[:, 2 : Ny + 2, 1]
                    - 2 * u2[:, 1 : Ny + 1, 1]
                    + u2[:, 0:Ny, 1]
                )
            )

            # corners
            u2[:, -1, 0] = a * (
                u1[:, -1, 0]
                - u2[:, Ny, 0]
                + u1[:, Ny, 0]
                + lambda_v * (u2[:, Ny, 0] - u1[:, -1, 0] + u1[:, Ny, 0])
            )
            u2[:, 0, 0] = a * (
                u1[:, 0, 0]
                - u2[:, 1, 0]
                + u1[:, 1, 0]
                + lambda_v * (u2[:, 1, 0] - u1[:, 0, 0] + u1[:, 1, 0])
            )
            u2[:, 0, -1] = a * (
                u1[:, 0, -1]
                - u2[:, 0, Nx]
                + u1[:, 0, Nx]
                + lambda_v * (u2[:, 0, Nx] - u1[:, 0, -1] + u1[:, 0, Nx])
            )
            u2[:, -1, -1] = a * (
                u1[:, -1, -1]
                - u2[:, Ny, -1]
                + u1[:, Ny, -1]
                + lambda_v * (u2[:, Ny, -1] - u1[:, -1, -1] + u1[:, Ny, -1])
            )

            # update grids
            u, ut = u2.clone(), (u2 - u0) / (2 * dt)
            u0 = u1.clone()
            u1 = u2.clone()
            Ny, Nx = Ny + 1, Nx + 1

        return u, ut

    else:
        raise NotImplementedError("this boundary condition is not implemented")


def pseudo_spectral_tensor(u0, ut0, vel, dx, dt, delta_t_star):
    """
    Parameters
    ----------
    u0 : (pytorch tensor) physical wave component, displacement of wave
    ut0 : (pytorch tensor) physical wave component derived by t, velocity of wave
    vel : (pytorch tensor) velocity profile dependent on x_1 and x_2
    dx : (float) time step in both dimensions / grid spacing
    dt : (float) temporal step size
    delta_t_star : (float) time step a solver propagates a wave and solvers are compared

    Returns
    -------
    propagate wavefield using RK4 in time and spectral approx. of Laplacian in space (batched)
    """

    def _spectral_del_tensor(v, dx):
        """

        Parameters
        ----------
        v : (pytorch tensor) velocity profile dependent on x_1 and x_2
        dx : (float) spatial step size / grid spacing (in x_1 and x_2 dimension)

        Returns
        -------
        evaluate the discrete Laplacian using spectral method (batched)
        """

        N1 = v.shape[-2]
        N2 = v.shape[-1]

        kx = (
            2
            * torch.pi
            / (dx * N1)
            * torch.fft.fftshift(torch.linspace(-round(N1 / 2), round(N1 / 2 - 1), N1))
        )
        ky = (
            2
            * torch.pi
            / (dx * N2)
            * torch.fft.fftshift(torch.linspace(-round(N2 / 2), round(N2 / 2 - 1), N2))
        )
        [kxx, kyy] = torch.meshgrid(kx, ky, indexing="xy")

        U = -(kxx**2 + kyy**2) * torch.fft.fft2(v)

        return torch.fft.ifft2(U)

    Nt = round(abs(delta_t_star / dt))
    c2 = torch.multiply(vel, vel)

    u = u0
    ut = ut0

    for i in range(Nt):
        # RK4 scheme
        k1u = ut
        k1ut = torch.multiply(c2, _spectral_del_tensor(u, dx))

        k2u = ut + dt / 2 * k1ut
        k2ut = torch.multiply(c2, _spectral_del_tensor(u + dt / 2 * k1u, dx))

        k3u = ut + dt / 2 * k2ut
        k3ut = torch.multiply(c2, _spectral_del_tensor(u + dt / 2 * k2u, dx))

        k4u = ut + dt * k3ut
        k4ut = torch.multiply(c2, _spectral_del_tensor(u + dt * k3u, dx))

        u = u + 1.0 / 6 * dt * (k1u + 2 * k2u + 2 * k3u + k4u)
        ut = ut + 1.0 / 6 * dt * (k1ut + 2 * k2ut + 2 * k3ut + k4ut)

    return torch.real(u), torch.real(ut)


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


def WaveEnergyField_tensor(u, ut, c, dx):
    """
    Parameters
    ----------
    u : (pytorch tensor) physical wave component, displacement of wave
    ut : (pytorch tensor) physical wave component derived by t, velocity of wave
    c : (pytorch tensor) velocity profile dependent on x_1 and x_2
    dx : (flaot) time step in both dimensions / grid spacing

    Returns
    -------
    transformation to energy-semi norm of wave given parameters c and dx (tensor)
    """

    ux, uy = torch.gradient(u, spacing=dx)
    absux = torch.abs(ux)
    absuy = torch.abs(uy)
    absutc = torch.divide(torch.abs(ut), c)
    w = (
        torch.multiply(absux, absux)
        + torch.multiply(absuy, absuy)
        + torch.multiply(absutc, absutc)
    )

    return w