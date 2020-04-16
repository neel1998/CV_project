#!/bin/bash
#SBATCH -w gnode53
#SBATCH -A neel1998
#SBATCH -n 10
#SBATCH --gres=gpu:0
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END

python3 ~/CV_project/src/create_test_data.py
