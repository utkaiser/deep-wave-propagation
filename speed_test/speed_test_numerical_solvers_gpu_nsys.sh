#!/bin/bash

#SBATCH --job-name=wave-speed-test-numerical
#SBATCH --output=outs/results_numerical_gpu_nvtx.out
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --partition=h100

pwd; hostname; date

module load nvidia/24.5
module load cuda/12.4

echo "--- Experiment 1: pseudo-spectral periodic ---"
nsys profile -t cuda,nvtx --stats=true -o outs/experiment1_nsys_report --force-overwrite true python3 speed_test/speed_test_numerical_solvers_nsys.py pseudo-spectral periodic
echo

echo "--- Experiment 2: velocity-verlet periodic ---"
nsys profile -t cuda,nvtx --stats=true -o outs/experiment2_nsys_report --force-overwrite true python3 speed_test/speed_test_numerical_solvers_nsys.py velocity-verlet periodic
echo

echo "--- Experiment 3: velocity-verlet absorbing ---"
nsys profile -t cuda,nvtx --stats=true -o outs/experiment3_nsys_report --force-overwrite true python3 speed_test/speed_test_numerical_solvers_nsys.py velocity-verlet absorbing
echo

# TODO: use experiment1_nsys_report.qdrep to see the report
