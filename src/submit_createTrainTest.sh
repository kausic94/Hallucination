#!/bin/bash
SBATCH -N 1
SBATCH -n 4
SBATCH -p cidsegpu2
SBATCH -q cidsegpu2
module load opencv/3.4.1
python createTrainTest.py --input /home/kgunase3/data/NYUD/RAW/Depth/ --groundTruth /home/kgunase3/data/NYUD/RAW/RGB/ --savePath /home/kgunase3/data/NYUD/RAW/

