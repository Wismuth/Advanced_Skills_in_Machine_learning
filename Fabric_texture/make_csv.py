import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.misc
import random
import csv


crop_img_path = "D:/Fabric_texture/data/"
number_of_classes = 0

with open('fabric.csv', 'a', newline='') as csvfile:
	for i in os.listdir(crop_img_path):
		crop_img_class_path = os.path.join(crop_img_path, i)
		img_class_number, class_name = os.path.splitext(i)
		for j in os.listdir(crop_img_class_path):
			filename = i + '/' + j
			csvwriter = csv.writer(csvfile)
			csvwriter.writerow([filename, img_class_number])
		#if number_of_classes == 100:
		#	break

