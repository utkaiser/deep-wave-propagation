#!/bin/bash

#SBATCH --job-name=wave-speed-test-numerical
#SBATCH --output=outs/results_numerical_cpu.out
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=skx

pwd; hostname; date

echo "--- Experiment 1: pseudo-spectral periodic ---"
python3 speed_test/speed_test_numerical_solvers.py pseudo-spectral periodic
echo

echo "--- Experiment 2: velocity-verlet periodic ---"
python3 speed_test/speed_test_numerical_solvers.py velocity-verlet periodic
echo

echo "--- Experiment 3: velocity-verlet absorbing ---"
python3 speed_test/speed_test_numerical_solvers.py velocity-verlet absorbing
echo
