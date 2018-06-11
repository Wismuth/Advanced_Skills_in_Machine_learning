import sys
import os
import csv
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim.nets
import cv2 as cv

# import googlenet

slim = tf.contrib.slim
inception = slim.nets.inception

input_width  = 224
input_height = 224
c_dim = 3

y_dim = 10

num_classes = y_dim

train_filename = 'fabric_train.csv'
valid_filename = 'fabric_valid.csv'

train_basepath = './data/'
valid_basepath = './data/'

#vis_basepath = './vis'

snapshot_basepath = './snapshot'

train_result = 'fabric_train_result.csv'
valid_result = 'fabric_valid_result.csv'

def preprocess(filename, basepath): 
  # print(filepath)
  img = cv.imread(filename)
  #print(filename)
  img = cv.resize(img, dsize=(input_height, input_width))
  fimg = img.astype(np.float32) / 255.0
  fimg = fimg - fimg.mean()
  fimg = fimg.reshape(1, input_width, input_height, c_dim)

  return fimg

def main(_):  
  train_data = []
  with open(train_filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:    
    #  if row[0][0] == 'I':
      filepath = os.path.join(train_basepath, row[0])
    #  else:
    #    filepath = os.path.join(train_basepath, row[0]+'.png')
      room = float(row[1])
      train_data.append([filepath, room])

  valid_data = []
  with open(valid_filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:  
    #  if row[0][0] == 'I':
      filepath = os.path.join(valid_basepath, row[0])
      #else:
      #  filepath = os.path.join(valid_basepath, row[0]+'.png')
      room = float(row[1])
      valid_data.append([filepath, room])

  print('number of train files', len(train_data), 
    'number of valid files', len(valid_data))

  print('prepare place holder')
  x_input = tf.placeholder(tf.float32, [None, input_height, input_width, c_dim], name='x_input')  
  
  print('define network')  
  with slim.arg_scope(inception.inception_v1_arg_scope()):
    y_conv, end_points = inception.inception_v1(x_input, 
      num_classes=num_classes, 
      is_training=False, 
      dropout_keep_prob=1.0)

  print('initialize session')
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config = config) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    print('Restoring model weights')
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(snapshot_basepath))

    print('run training testing')    
    with open(train_result, 'w', newline=''):
      pass

    for data in train_data:
      filename = data[0]
      room = data[1]

      img = preprocess(filename, train_basepath)

      y_ = sess.run(y_conv, feed_dict={x_input:img})

      with open(train_result, 'a', newline='') as file:   
        csvwriter = csv.writer(file) 
        csvwriter.writerow([filename, 
          room, np.argmax(y_)])

    print('run validation testing')    
    with open(valid_result, 'w', newline=''):
      pass

    for data in valid_data:
      filename = data[0]
      img = preprocess(filename, valid_basepath)
      room = data[1]

      y_ = sess.run(y_conv, feed_dict={x_input:img})
      max_y = max(y_)
      with open(valid_result, 'a', newline='') as file:   
        csvwriter = csv.writer(file) 
        csvwriter.writerow([filename, 
          room, np.argmax(y_)+1])

    
  
if __name__ == '__main__':
  tf.app.run(main=main)
