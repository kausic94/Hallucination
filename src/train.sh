#!/bin/bash 
python3 -u RGBHallucinator.py 2  0 &
python3 -u RGBHallucinator.py 4  1 &
python3 -u RGBHallucinator.py 8  2 &
python3 -u RGBHallucinator.py 16 2 &

wait 
echo "All training scripts have completed succcessfully"
