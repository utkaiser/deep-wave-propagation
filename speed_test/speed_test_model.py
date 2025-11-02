import argparse
import torch
import time
import sys
import os

import warnings
warnings.filterwarnings("ignore")

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_params
from model_old import Model_end_to_end as Model_end_to_end_old
from model import Model_end_to_end
from generate_data import fetch_data_end_to_end


def test_speed_of_model(
    upsampling_model = "UNet3",
    data_paths = "data/datagen_test2.npz",
    val_paths = "data/datagen_test2.npz",
    batch_size = 1,
    tolerance = 1e-6,
    num_decimals_round = 6,
    n_trials = 10,
):
    """
    Parameters
    ----------
    downsampling_model : (string) name of downsampling model
    upsampling_model : (string) name of upsampling model
    data_paths : (string)
    val_paths : (string)
    batch_size : (int) number of samples per batch
    num_decimals_round : (int) number of decimals to round speed results to
    n_trials : (int) number of trials to run

    Returns
    ----------
    Speed test of end-to-end model for advancing wave fields
    """

    # model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    param_dict = get_params()  # get dict of params contains all model specifications such as numerical params
    model_old = Model_end_to_end_old(param_dict, upsampling_model)
    model_old = torch.nn.DataParallel(model_old).to(device)  # multi-GPU use
    model = Model_end_to_end(param_dict, upsampling_model)
    model = torch.nn.DataParallel(model).to(device)  # multi-GPU use
    model_compiled = Model_end_to_end(param_dict, upsampling_model)
    model_compiled = torch.nn.DataParallel(model_compiled).to(device)  # multi-GPU use
    model_compiled = torch.compile(model_compiled)

    # warmup run for compiled model
    print("Starting warmup run for compiled model...")
    start_time_warmup = time.perf_counter()
    model_compiled.eval()
    with torch.inference_mode():
        for _ in range(5):
            dummy_input = torch.randn(batch_size, 5, 128, 128).to(device)
            dummy_vel = torch.randn(batch_size, 128, 128).to(device)
            _ = model_compiled(dummy_input, dummy_vel)
    end_time_warmup = time.perf_counter()
    print(f"Warmup run for compiled model took {round(end_time_warmup - start_time_warmup, num_decimals_round)} seconds.")

    # data setup
    data_loader, _, _ = fetch_data_end_to_end([data_paths], batch_size, [val_paths])

    # set up lists to store results
    speed_results_old, speed_results_optimized, speed_results_compiled, speed_results_compiled_autocast, interations_all_close = [], [], [], [], []

    model.eval()
    model_old.eval()
    model_compiled.eval()
    with torch.inference_mode():
        for trial in range(n_trials):
            for i, data_init in enumerate(data_loader):  # iterate over datapoints in val_loader
                if i > 0: break

                speed_results_old.append([]), speed_results_optimized.append([]), speed_results_compiled.append([]), speed_results_compiled_autocast.append([]), interations_all_close.append([])
                n_snaps = data_init[0].shape[1]  # number of snapshots defined by data input

                # input tensor setup
                data_old = data_init[0].clone().to(device)  # use GPUs if available for faster execution
                input_tensor_old = data_old[:, 0].detach()  # detach because if not computation graph would go too far back
                vel_old = input_tensor_old[:, 3].unsqueeze(dim=1)  # access velocity profile in input_tensor
                
                data_opt = data_init[0].clone().to(device)  # use GPUs if available for faster execution
                input_tensor_opt = data_opt[:, 0].detach()  # detach because if not computation graph would go too far back
                vel_opt = input_tensor_opt[:, 3].unsqueeze(dim=1)  # access velocity profile in input_tensor

                data_compiled = data_init[0].clone().to(device)  # use GPUs if available for faster execution
                input_tensor_compiled = data_compiled[:, 0].detach()  # detach because if not computation graph would go too far back
                vel_compiled = input_tensor_compiled[:, 3].unsqueeze(dim=1)  # access velocity profile in input_tensor

                data_compiled_autocast = data_init[0].clone().to(device)  # use GPUs if available for faster execution
                input_tensor_compiled_autocast = data_compiled_autocast[:, 0].detach()  # detach because if not computation graph would go too far back
                vel_compiled_autocast = input_tensor_compiled_autocast[:, 3].unsqueeze(dim=1)  # access velocity profile in input_tensor

                for s in range(1, n_snaps):  # advance a wave field for (n_snaps - 1) time steps
                    # old implementation speed test
                    start_time_old = time.perf_counter()
                    output_old = model_old(input_tensor_old, data_old[:, 0, 4])  # apply end-to-end model
                    end_time_old = time.perf_counter()

                    # optimized implementation speed test
                    start_time_opt = time.perf_counter()
                    output_opt = model(input_tensor_opt, data_opt[:, 0, 4])  # apply end-to-end model
                    end_time_opt = time.perf_counter()
                    
                    # compiled implementation speed test
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        start_time_compiled = time.perf_counter()
                        output_compiled = model_compiled(input_tensor_compiled, data_compiled[:, 0, 4])  # apply end-to-end model
                        end_time_compiled = time.perf_counter()

                    # compiled + autocast implementation speed test
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        start_time_compiled_autocast = time.perf_counter()
                        output_compiled_autocast = model_compiled(input_tensor_compiled, data_compiled[:, 0, 4])  # apply end-to-end model
                        end_time_compiled_autocast = time.perf_counter()

                    # update input tensors for next iteration
                    input_tensor_old = torch.cat((output_old, vel_old), dim=1)
                    input_tensor_opt = torch.cat((output_opt, vel_opt), dim=1)
                    input_tensor_compiled = torch.cat((output_compiled, vel_compiled), dim=1)
                    input_tensor_compiled_autocast = torch.cat((output_compiled_autocast, vel_compiled_autocast), dim=1)

                    # store results
                    speed_results_old[trial].append(round(end_time_old - start_time_old, num_decimals_round))
                    speed_results_optimized[trial].append(round(end_time_opt - start_time_opt, num_decimals_round))
                    speed_results_compiled[trial].append(round(end_time_compiled - start_time_compiled, num_decimals_round))
                    speed_results_compiled_autocast[trial].append(round(end_time_compiled_autocast - start_time_compiled_autocast, num_decimals_round))
                    interations_all_close[trial].append(torch.allclose(output_old, output_opt, atol=tolerance, rtol=0.) and torch.allclose(output_old, output_compiled, atol=tolerance, rtol=0.) and torch.allclose(output_opt, output_compiled, atol=tolerance, rtol=0.))

    print(f"Old implementation: {speed_results_old}.")
    print(f"Optimized implementation: {speed_results_optimized}.")
    print(f"Compiled implementation: {speed_results_compiled}.")
    print(f"Compiled with autocast implementation: {speed_results_compiled_autocast}.")
    print(f"All close per iteration: {interations_all_close}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run speed test experiments with specified model architecture.")
    parser.add_argument('upsampling_model', type=str, help="Upsampling model name")
    parser.add_argument('batch_size', type=int, help="Batch size for data loader")
    args = parser.parse_args()
    upsampling_model, batch_size = args.upsampling_model, args.batch_size

    print(f"Running experiment with {upsampling_model=}, {batch_size=}")
    test_speed_of_model(
        upsampling_model=upsampling_model,
        batch_size=batch_size
    )
