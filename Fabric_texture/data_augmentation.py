import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.misc
import random

window_width = 224
window_height = 224


def img_read(path):
    #print(path)
    #img = scipy.misc.imread(path).astype(np.float)
    img = cv.imread(path).astype(np.float)
    #print(img.shape[:2])
    #image resie 256 x 256
    #img = cv.resize(img, (resize_scale, resize_scale))
    if len(img.shape) == 2:
        img = np.dstack((img, img, img))
    elif img.shape[2] == 4:
        img = img[:,:,:3]
    return img

def img_save(path, img):
    #img = np.clip(img, 0, 255).astype(np.uint8)
    cv.imwrite(path, img)

origin_img_dir_path = "./origin"
data_img_dir_path = "./data"
if not os.path.exists(data_img_dir_path):
    os.makedirs(data_img_dir_path)

for i in os.listdir(origin_img_dir_path):
    origin_class_path = os.path.join(origin_img_dir_path, i)
    data_class_path = os.path.join(data_img_dir_path, i)

    class_number, class_name = i.split('.')
    print(class_name)
    if not os.path.exists(data_class_path):
        os.makedirs(data_class_path)

    img_number = 1
    for j in os.listdir(origin_class_path):
        origin_full_path = os.path.join(origin_class_path, j)
        print(origin_full_path)
        origin_img = img_read(origin_full_path)
        x_list = []
        y_list = []
        for k in range(6000):
            y = random.randint(0, origin_img.shape[0] - window_width - 1)
            x = random.randint(0, origin_img.shape[1] - window_height - 1)
            while(x not in x_list and y not in y_list):
                y = random.randint(0, origin_img.shape[0] - window_width - 1)
                x = random.randint(0, origin_img.shape[1] - window_height - 1)
                x_list.append(x)
                y_list.append(y)
            crop_img = origin_img[y:y+224, x:x+224]
            crop_filename = class_name+"_" + str(img_number) + ".jpg"
            crop_fullpath = os.path.join(data_class_path, crop_filename)
            img_save(crop_fullpath, crop_img)
            img_number += 1