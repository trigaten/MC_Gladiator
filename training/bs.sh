#!/bin/sh
#BATCH --time=20:00:00
#SBATCH --partition=gpu
#SBATCH --qos=gpu-medium
#SBATCH --gres=gpu:1
#SBATCH --ntasks=5
####SBATCH --output /cliphomes/sschulho/out.txt
####SBATCH --error /cliphomes/sschulho/out.txt
echo $(hostname)
xvfb-run python3 marl.py
