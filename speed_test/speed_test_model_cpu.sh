#!/bin/bash

#SBATCH --job-name=wave-speed-test-model
#SBATCH --output=outs/results_model_cpu.out
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=skx


pwd; hostname; date

echo "--- Experiment 1: UNet3 1 ---"
python3 speed_test/speed_test_model.py UNet3 1
echo

echo "--- Experiment 2: UNet3 16 ---"
python3 speed_test/speed_test_model.py UNet3 16
echo

echo "--- Experiment 3: UNet3 128 ---"
python3 speed_test/speed_test_model.py UNet3 128
echo

echo "--- Experiment 4: UTransformer_old 1 ---"
python3 speed_test/speed_test_model.py UTransformer_old 1
echo

echo "--- Experiment 5: UTransformer_old 16 ---"
python3 speed_test/speed_test_model.py UTransformer_old 16
echo

echo "--- Experiment 6: UTransformer_old 128 ---"
python3 speed_test/speed_test_model.py UTransformer_old 128
echo

echo "--- Experiment 7: UTransformer 1 ---"
python3 speed_test/speed_test_model.py UTransformer 1
echo

echo "--- Experiment 8: UTransformer 16 ---"
python3 speed_test/speed_test_model.py UTransformer 16
echo

echo "--- Experiment 9: UTransformer 128 ---"
python3 speed_test/speed_test_model.py UTransformer 128
echo