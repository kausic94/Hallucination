#!/usr/bin/env python
# coding: utf-8
import numpy as np
import cv2
import argparse
import os
import sys
from tqdm import tqdm
from sklearn.utils import shuffle
parser=argparse.ArgumentParser(description="Helps to create a train test split of the data you have")
parser.add_argument('--input',help='directory that contains all your input data (depth images) ',default='./')
parser.add_argument('--groundTruth',help='directory that contains your ground truth (RGB images)',default='./')
parser.add_argument('--splitRatio', type = float , help = 'what portion of the data do you want as train 0>ratio>1',default='0.8')
parser.add_argument('--savePath',help='Directory where you want the train and test files to be created',default='./')
args=parser.parse_args()
inputDir=args.input
gtDir = args.groundTruth
splitRatio=args.splitRatio
savePath=args.savePath
rgb=os.listdir(gtDir)
depth=os.listdir(inputDir)
rgb.sort()
depth.sort()
assert len(rgb) == len(depth), "Input and ground truth directories don't have equal number of images"
depthNames,RGBNames = [] , []
count=0
for i in tqdm(range(len(rgb)),desc="Check and append"):
    rgbn=os.path.join(gtDir,rgb[i])
    depthn=os.path.join(inputDir,depth[i])
    if cv2.imread(rgbn) is None or cv2.imread(depthn) is None :
        continue    
    depthNames.append(depthn)
    RGBNames.append(rgbn)
    

depthNames,RGBNames= shuffle(depthNames,RGBNames)
index= int(len(depthNames)*splitRatio)
train_d,train_rgb = depthNames[:index],RGBNames[:index]
test_d,test_rgb = depthNames[index:],RGBNames[index:]
train_file,test_file=os.path.join(savePath,"train.txt"),os.path.join(savePath,"test.txt")
with open(train_file,'w') as fh:
    for i in tqdm(range(len(train_d)),desc="Preparing train files"):
        fh.write(train_d[i] + ',' + train_rgb[i] + '\n')
with open(test_file,'w') as fh:
    for i in tqdm(range(len(test_d)),desc="Preparing test files"):
        fh.write(test_d[i] + ',' + test_rgb[i] + '\n')
