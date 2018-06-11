import sys
import os
import csv
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim.nets
import cv2 as cv
import random

slim = tf.contrib.slim
inception = slim.nets.inception

input_width  = 224
input_height = 224
c_dim = 3

snapshot_basepath = './snapshot'

def square_crop(img):
  shape = img.shape[:2] 
  min_side = min(shape)

  start=((shape[0] - min_side)//2, (shape[1] - min_side)//2)

  croped_img = img[start[0]:start[0]+min_side, start[1]:start[1]+min_side]
  
  return croped_img

def preprocess(img): 
  croped_img = square_crop(img)
  resized_img = cv.resize(croped_img, dsize=(input_height, input_width))

  fimg = resized_img.astype(np.float32) / 255.0
  fimg = fimg - fimg.mean()
  fimg = fimg.reshape(1, input_width, input_height, c_dim)

  return fimg

def softmax(x):
  theta = 1.0
  ex = np.exp(theta*x)
  ex /= np.sum(ex)

  return ex


def main():  
  print('prepare place holder')
  x_input = tf.placeholder(tf.float32, [None, input_height, input_width, c_dim], name='x_input')  
  
  print('define network')  
  with slim.arg_scope(inception.inception_v1_arg_scope()):
    y_conv, end_points = inception.inception_v1(x_input, 
      num_classes=10, 
      is_training=False, 
      dropout_keep_prob=1.0)

  locations = ['abstract','circlepattern', 'leaf', 'floral', 'swirling', 'blocks', 'animal', 'nopattern', 'dotswirl', 'another']

  print('initialize session')  
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
        
    print('Restoring model weights')
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(snapshot_basepath))

    print('run testing')
    img_basepath = './data/'
    for (dirpath, dirnames, filenames) in os.walk(img_basepath):
      random.shuffle(filenames)
      for filename in filenames:
        img_fullpath = os.path.join(dirpath, filename)
        img = cv.imread(img_fullpath)
        scale = 320.0 / img.shape[1] 
        img = cv.resize(img, dsize=(0,0), fx=scale, fy=scale)
        cv.imshow('input img', img)
        cv.waitKey(30)
        
        fimg = preprocess(img)
        y_ = sess.run(y_conv, feed_dict={x_input:fimg})

        idx = np.argmax(y_[0])

        font = cv.FONT_HERSHEY_SIMPLEX        
        cv.putText(img, locations[idx], (10,30), font, 1, (0,255,255), 2, cv.LINE_AA)
        
        prob = softmax(y_[0]) * 250
        scores = np.ones([250, 250, 3], np.uint8)
        scores = scores * 255
        cv.rectangle(scores, (  0,250), (25,250-int(prob[0])), (0,0,255), -1)
        cv.rectangle(scores, (  25,250), (50,250-int(prob[1])), (0,0,255), -1)
        cv.rectangle(scores, (  50,250), (75,250-int(prob[2])), (0,0,255), -1)
        cv.rectangle(scores, (  75,250), (100,250-int(prob[3])), (0,0,255), -1)
        cv.rectangle(scores, (  100,250), (125,250-int(prob[4])), (0,0,255), -1)
        cv.rectangle(scores, (  125,250), (150,250-int(prob[5])), (0,0,255), -1)
        cv.rectangle(scores, (  150,250), (175,250-int(prob[6])), (0,0,255), -1)
        cv.rectangle(scores, (  175,250), (200,250-int(prob[7])), (0,0,255), -1)
        cv.rectangle(scores, (  200,250), (225,250-int(prob[8])), (0,0,255), -1)
        cv.rectangle(scores, (  225,250), (250,250-int(prob[9])), (0,0,255), -1) 

        cv.imshow('scores', scores)
        cv.imshow('fabric img', img)
        cv.waitKey(300)
        
    

if __name__ == '__main__':
  main()
