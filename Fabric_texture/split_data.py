import sys
import os
import numpy as np
import cv2 as cv
import h5py
import math
import csv

metafile_converted = 'fabric.csv'

metafile_train = 'fabric_train.csv'
metafile_valid = 'fabric_valid.csv'

def main(argv):
  with open(metafile_converted, 'r') as file:
    lines = file.readlines()    
    print(lines[0])
  num_data = len(lines)
  for i in range(num_data):
    if i % 10 != 0:
      with open(metafile_train,'a') as f:
        f.write(lines[i])
    else:
      with open(metafile_valid,'a') as f:
        f.write(lines[i])
if __name__ == '__main__':
  main(sys.argv)


