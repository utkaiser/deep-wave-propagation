import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# --- OPTIMIZATION: Helper function uses pre-computed operator ---
@torch.jit.script
def _spectral_del_tensor_fast(v, laplacian_op_k_space, N1: int, N2: int):
    """
    Evaluate the discrete Laplacian using pre-computed spectral operator.
    Uses rfft2/irfft2 for real-to-real transform.
    """
    # 1. Transform to k-space (real-to-complex)
    v_k = torch.fft.rfft2(v)
    
    # 2. Apply Laplacian operator (element-wise multiplication)
    U = laplacian_op_k_space * v_k
    
    # 3. Transform back to spatial domain (complex-to-real)
    #    We must specify the original size (s=) for irfft2
    return torch.fft.irfft2(U, s=(N1, N2))


# --- OPTIMIZATION: Helper function uses pre-computed operator ---
@torch.jit.script
def _periLaplacian_tensor(v, dx: float, number: int):
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
    # --- OPTIMIZATION: Combined terms for fewer operations ---
    dx2 = dx**2

    Lv = (
        torch.roll(v, 1, dims=1 + number)
        - 2 * v
        + torch.roll(v, -1, dims=1 + number)
    ) / dx2 + (
        torch.roll(v, 1, dims=0 + number)
        - 2 * v
        + torch.roll(v, -1, dims=0 + number)
    ) / dx2

    return Lv


# --- OPTIMIZATION: Added JIT decorator ---
@torch.jit.script
def velocity_verlet_tensor(
    u0, ut0, vel, dx: float, dt: float, delta_t_star: float, number: int=0, boundary_c: str ="periodic"
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

    # --- OPTIMIZATION: make it a true int ---
    Nt = int(round(abs(delta_t_star / dt)))
    # --- OPTIMIZATION: Use * operator instead of torch.mul ---
    u, ut = u0, ut0

    if boundary_c == "periodic":
        c2 = vel * vel
        for i in range(Nt):
            # --- OPTIMIZATION: Use autocast for mixed precision ---
            with torch.cuda.amp.autocast():
                ddxou = _periLaplacian_tensor(u, dx, number)
                u = u + dt * ut + 0.5 * dt**2 * torch.mul(c2, ddxou)
                ddxu = _periLaplacian_tensor(u, dx, number)
            ut = ut + 0.5 * dt * torch.mul(c2, ddxou + ddxu)

        return u, ut

    elif boundary_c == "absorbing":
        # shape: u, ut -> b x w_c x h_c
        # --- Handle 2D or 3D input ---
        needs_squeeze = False
        if u0.ndim == 2:
            # Add a batch dimension of 1
            u0 = u0.unsqueeze(0)
            ut0 = ut0.unsqueeze(0)
            vel = vel.unsqueeze(0)  # 'vel' is also used with 3D slicing (via c2/lambdaC2)
            needs_squeeze = True

        # --- OPTIMIZATION: Ensure shape indices are integers for JIT ---
        Ny = u0.shape[-1] - 1
        Nx = u0.shape[-2] - 1

        lambda_v = abs(dt / dx)
        lambda2 = lambda_v**2
        c2 = vel * vel
        lambdaC2 = lambda2 * c2

        a = dx / (dx + abs(dt))

        # Euler step to generate u1 from u0 and ut0
        uneg1 = u0 - dt * ut0
        u2 = u0.clone()
        u1 = u0.clone()
        u0 = uneg1.clone()

        for k in range(Nt):

            # --- OPTIMIZATION: Use autocast for mixed precision ---
            with torch.cuda.amp.autocast():
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

        if needs_squeeze:
            u = u.squeeze(0)
            ut = ut.squeeze(0)
        return u, ut

    else:
        raise NotImplementedError("this boundary condition is not implemented")


# --- OPTIMIZATION: Added JIT decorator ---
@torch.jit.script
def pseudo_spectral_tensor(u0, ut0, vel, dx: float, dt: float, delta_t_star: float):
    """
    Propagate wavefield using RK4 in time and spectral approx. of Laplacian in space (batched).
    """
    # --- OPTIMIZATION: Make it a true int ---
    Nt = int(round(abs(delta_t_star / dt)))
    # --- OPTIMIZATION: Use * operator ---
    c2 = vel * vel

    u = u0
    ut = ut0
    
    # --- OPTIMIZATION: Pre-compute Laplacian operator and use rfft ---
    N1 = u0.shape[-2]
    N2 = u0.shape[-1]
    
    # Create k-vectors on the same device and dtype as input
    # Use rfft frequencies for ~2x speedup
    kx = (
        2 * torch.pi / (dx * N1)
        * torch.fft.fftshift(
            torch.linspace(-round(N1 / 2), round(N1 / 2 - 1), N1, device=u0.device, dtype=u0.dtype)
        )
    )
    # k-vector for last dimension (rfft)
    ky = (
        2 * torch.pi / (dx * N2)
        * torch.linspace(0, round(N2 / 2), N2 // 2 + 1, device=u0.device, dtype=u0.dtype)
    )

    # --- OPTIMIZATION: Remove square brackets for compilation ---
    kxx, kyy = torch.meshgrid(kx, ky, indexing="ij")
    
    # This is the Laplacian operator in k-space.
    # It has shape [..., N1, N2//2 + 1]
    laplacian_op_k_space = -(kxx**2 + kyy**2)

    # --- OPTIMIZATION: Make broadcasting explicit for JIT by adding a batch dim ---
    # Shape changes from [128, 65] to [1, 128, 65]
    laplacian_op_k_space_batched = laplacian_op_k_space.unsqueeze(0)

    for i in range(Nt):
        # --- OPTIMIZATION: Use autocast for mixed precision ---
        with torch.cuda.amp.autocast():
            # RK4 scheme
            k1u = ut
            k1ut = c2 * _spectral_del_tensor_fast(u, laplacian_op_k_space_batched, N1, N2)

            k2u = ut + dt / 2 * k1ut
            k2ut = c2 * _spectral_del_tensor_fast(u + dt / 2 * k1u, laplacian_op_k_space_batched, N1, N2)

            k3u = ut + dt / 2 * k2ut
            k3ut = c2 * _spectral_del_tensor_fast(u + dt / 2 * k2u, laplacian_op_k_space_batched, N1, N2)

            k4u = ut + dt * k3ut
            k4ut = c2 * _spectral_del_tensor_fast(u + dt * k3u, laplacian_op_k_space_batched, N1, N2)

        u = u + (1.0 / 6.0) * dt * (k1u + 2 * k2u + 2 * k3u + k4u)
        ut = ut + (1.0 / 6.0) * dt * (k1ut + 2 * k2ut + 2 * k3ut + k4ut)

    # --- OPTIMIZATION: No need for torch.real() as irfft2 returns real tensor ---
    return u, ut
