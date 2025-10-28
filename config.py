def get_params():
    """
    Parameters
    ----------

    Returns
    -------
    (dictionary) get numerical and training parameters
    """

    d = {
        "n_epochs": 20,
        "n_snaps": 8,
        "boundary_c": "absorbing",
        "delta_t_star": 0.06,
        "f_delta_x": 2.0 / 128.0,
        "f_delta_t": (2.0 / 128.0) / 20.0,
        "c_delta_x": 2.0 / 64.0,
        "c_delta_t": 1.0 / 600.0,
        "optimizer_name": "AdamW",
        "loss_function_name": "MSE",
        "res_scaler": 2,
    }

    return d