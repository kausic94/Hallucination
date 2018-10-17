#!/bin/bash
#SBATCH -N 1
#SBATCH -n 5
#SBATCH -p cidsegpu2
#SBATCH -q cidsegpu2
#SBATCH -t 7-00:00
#SBATCH --job-name SMOOTH_RMSE_linknet
#SBATCH --gres=gpu:V100:1
module load opencv/3.4.1 
module unload python/2.7.14
module load tensorflow/1.8-agave-gpu  
module list
python3 -u hallucinate.py config_test.ini 1 0
