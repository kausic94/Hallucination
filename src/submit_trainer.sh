#!/bin/bash
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -p cidsegpu2
#SBATCH -q cidsegpu2
#SBATCH --gres=gpu:V100:3
module load opencv/3.4.1 
module unload python/2.7.14
module load tensorflow/1.8-agave-gpu  
module list
./train.sh
