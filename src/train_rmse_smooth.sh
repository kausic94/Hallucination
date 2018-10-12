#!/bin/bash 
python3 -u hallucinate.py config_smooth_rmse.ini 2  0 &
python3 -u hallucinate.py config_smooth_rmse.ini 4  1 &
python3 -u hallucinate.py config_smooth_rmse.ini 8  2 &
python3 -u hallucinate.py config_smooth_rmse.ini 16 2 &

wait 
echo "All training scripts have completed succcessfully"
