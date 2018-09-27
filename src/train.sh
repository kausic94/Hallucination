#!/bin/bash 
python RGBHallucinator.py 2  0 &
python RGBHallucinator.py 4  1 &
python RGBHallucinator.py 8  2 &
python RGBHallucinator.py 16 2 &

wait 
echo "All training scripts have completed succcessfully"
