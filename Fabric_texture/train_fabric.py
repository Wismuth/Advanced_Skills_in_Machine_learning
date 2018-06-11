import h5py
import sys
import os
import csv
import threading
import random
import numpy as np
import numpy.linalg as LA
import tensorflow as tf
import tensorflow.contrib.slim.nets
import cv2 as cv

slim = tf.contrib.slim
inception = slim.nets.inception

max_iter = 200000
initial_learning_rate = 1e-5
beta1 = 0.9

decay_step = 2000
decay_rate = 0.96

batch_size = 64


input_width  = 224
input_height = 224
c_dim = 3

y_dim = 10
num_classes = y_dim

train_filename = 'fabric_train.csv'
valid_filename = 'fabric_valid.csv'

train_basepath = './data/'
valid_basepath = './data/'

snapshot_basepath = './snapshot'
snapshot_prefix = 'snapshot'
snapshot_interval = 5000

summary_basepath = './board'
summary_interval = 100

train_queue_capacity = 512
train_batch_capacity = 512

valid_queue_capacity = 128
valid_batch_capacity = 128

def enqueue(sess, stop_event, data, batch_size, enqueue_op, 
  queue_x_input, queue_y_input, tag='Train'):
  num_data = len(data)  
  data = np.array(data)
  print(tag, 'num data', data.shape)

  epoch = 0
  while True:
    print(tag, 'epoch', epoch)
    epoch = epoch + 1

    idxs = np.arange(0, num_data)
    np.random.shuffle(idxs)    

    shuf_data = data[idxs]

    for i in range(0, num_data-batch_size, batch_size):
      curr_x_input = []
      curr_y_input = []
      indices = []      
      for j in range(i, i+batch_size):
        #print(shuf_data[j][0])
        img = cv.imread(shuf_data[j][0])
        img = img.astype(np.float32) / 255.0
        x_input = img - img.mean()
        curr_x_input.append(x_input)
        #indices.append(int(float(shuf_data[j][1])))
        if int(float(shuf_data[j][1])) == 1:
          curr_y_input.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif int(float(shuf_data[j][1])) == 2:
          curr_y_input.append([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        elif int(float(shuf_data[j][1])) == 3:
          curr_y_input.append([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        elif int(float(shuf_data[j][1])) == 4:
          curr_y_input.append([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        elif int(float(shuf_data[j][1])) == 5:
          curr_y_input.append([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        elif int(float(shuf_data[j][1])) == 6:
          curr_y_input.append([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        elif int(float(shuf_data[j][1])) == 7:
          curr_y_input.append([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        elif int(float(shuf_data[j][1])) == 8:
          curr_y_input.append([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        elif int(float(shuf_data[j][1])) == 9:
          curr_y_input.append([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        else:
          curr_y_input.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
      #curr_y_input.append(tf.one_hot(indices, y_dim, on_value = 1.0, off_value = 0.0))
      curr_x_input = np.array(curr_x_input, dtype=np.float32)
      curr_y_input = np.array(curr_y_input, dtype=np.float32)
      sess.run(enqueue_op, 
        feed_dict={queue_x_input: curr_x_input, queue_y_input: curr_y_input})

      if stop_event.is_set():
        return

def main(_):  
  if not tf.gfile.Exists(snapshot_basepath):
    tf.gfile.MakeDirs(snapshot_basepath)

  train_data = []
  with open(train_filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
      #if row[0][0] == 'I':
      filepath = os.path.join(train_basepath, row[0])
      #else:
      #  filepath = os.path.join(train_basepath, row[0]+'.png')
      room = float(row[1])
      train_data.append([filepath, room])

  valid_data = []  
  with open(valid_filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
      #if row[0][0] == 'I':
      filepath = os.path.join(train_basepath, row[0])
      #else:
      #filepath = os.path.join(train_basepath, row[0]+'.png')
      room = float(row[1])
      valid_data.append([filepath, room])

  print('number of train files', len(train_data), 
    'number of valid files', len(valid_data))

  print('prepare queue')  
  # train queue
  train_queue = tf.FIFOQueue(capacity=train_queue_capacity, 
    dtypes=[tf.float32, tf.float32], 
    shapes=[[input_height, input_width, c_dim], [y_dim]])

  train_queue_x_input = tf.placeholder(tf.float32, shape=[None, input_height, input_width, c_dim])
  train_queue_y_input = tf.placeholder(tf.float32, shape=[None, y_dim])
  
  train_enqueue_op = train_queue.enqueue_many(
    [train_queue_x_input, train_queue_y_input])
  train_dequeue_op = train_queue.dequeue()
   
  train_batch_x_input, train_batch_y_input = \
    tf.train.batch(train_dequeue_op, batch_size=batch_size, capacity=train_batch_capacity)

  # validation queue
  valid_queue = tf.FIFOQueue(capacity=valid_queue_capacity, 
    dtypes=[tf.float32, tf.float32], 
    shapes=[[input_height, input_width, c_dim], [y_dim]])

  valid_queue_x_input = tf.placeholder(tf.float32, shape=[None, input_height, input_width, c_dim])
  valid_queue_y_input = tf.placeholder(tf.float32, shape=[None, y_dim])
  
  valid_enqueue_op = valid_queue.enqueue_many(
    [valid_queue_x_input, valid_queue_y_input])
  valid_dequeue_op = valid_queue.dequeue()

  valid_batch_x_input, valid_batch_y_input \
   = tf.train.batch(valid_dequeue_op, batch_size=batch_size, capacity=valid_batch_capacity)

  print('prepare place holder')
  x_input = tf.placeholder(tf.float32, [None, input_height, input_width, c_dim], name='x_input')  
  y_input = tf.placeholder(tf.float32, [None, y_dim], name='y_input')
  
  print('define network')
  is_training = tf.placeholder(tf.bool, name='is_training')
  dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

  # training network
  with slim.arg_scope(inception.inception_v1_arg_scope()):
    y_conv, end_points = inception.inception_v1(x_input, 
      num_classes=num_classes, is_training=is_training, dropout_keep_prob=dropout_keep_prob)

  # define loss
  #softmax
  #cross_entropy
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=y_conv))

  tf.summary.scalar('loss', loss)

  print('variables to restore:')
  variables_to_restore = slim.get_variables_to_restore(
      exclude=['InceptionV1/Logit', 'Variable', 'beta1_power', 'beta2_power'])
  for v in variables_to_restore:
    print(v.name)

  print('assign from checkpoint:') 
  init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
    './inception_v1.ckpt', variables_to_restore)

  print('define training operation:')
  global_step = tf.Variable(0, trainable=False)
  learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, 
                                            decay_step, decay_rate, staircase=True)
  
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss)
    clipped_grads = []
    for grad, var in grads_and_vars:
      clipped_grads.append((None if grad is None else tf.clip_by_value(grad, -1., 1.), var))   
    train_op = optimizer.apply_gradients(clipped_grads)

  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_input, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  
  tf.summary.scalar('room accuracy', accuracy)

  merged = tf.summary.merge_all()

  snapshot_path = os.path.join(snapshot_basepath, snapshot_prefix)

  print('initialize session')
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config = config) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    train_step_writer = tf.summary.FileWriter(summary_basepath+'/train_step', sess.graph)
    train_writer = tf.summary.FileWriter(summary_basepath+'/train')
    valid_writer = tf.summary.FileWriter(summary_basepath+'/valid')    

    print('Restoring model weights')
    #sess.run(init_assign_op, init_feed_dict)    

    saver = tf.train.Saver(max_to_keep=20)

    print('start enqueue threads')
    stop_event = threading.Event()

    train_enqueue_thread = threading.Thread(target=enqueue, 
      args=[sess, 
            stop_event, 
            train_data, 
            batch_size, 
            train_enqueue_op, 
            train_queue_x_input, 
            train_queue_y_input,
            'Train'])
    train_enqueue_thread.isDaemon()
    train_enqueue_thread.start()

    valid_enqueue_thread = threading.Thread(target=enqueue, 
      args=[sess, 
            stop_event, 
            valid_data, 
            batch_size, 
            valid_enqueue_op, 
            valid_queue_x_input, 
            valid_queue_y_input,
            'Valid'])
    valid_enqueue_thread.isDaemon()
    valid_enqueue_thread.start()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    
    print('run training')  
    for i in range(0, max_iter):
      train_x_batch, train_y_batch = \
        sess.run([train_batch_x_input, train_batch_y_input])

      loss_train_val = sess.run(train_op,
        feed_dict={x_input:train_x_batch, y_input:train_y_batch,
                is_training:True, dropout_keep_prob:0.8})

      if i%summary_interval == 0:
        train_step_summary = sess.run(merged, 
          feed_dict={x_input:train_x_batch, y_input:train_y_batch,
                    is_training:True, dropout_keep_prob:0.8})    

        train_summary, train_accuracy = sess.run([merged, accuracy], 
          feed_dict={x_input:train_x_batch, y_input:train_y_batch,
                    is_training:False, dropout_keep_prob:1.0})    
        
        valid_x_batch, valid_y_batch = \
          sess.run([valid_batch_x_input, valid_batch_y_input])
        
        valid_summary, valid_accuracy = sess.run([merged, accuracy], 
          feed_dict={x_input:valid_x_batch, y_input:valid_y_batch, 
                    is_training:False, dropout_keep_prob:1.0})

        train_step_writer.add_summary(train_step_summary, i)
        train_writer.add_summary(train_summary, i)
        valid_writer.add_summary(valid_summary, i)

        print('step', i, 
          'train accuracy', train_accuracy, 
          'valid accuracy', valid_accuracy)
        print('loss_train_val: ', loss_train_val)
      if i%snapshot_interval == 0:        
        saver.save(sess, snapshot_path, global_step=i)   

    saver.save(sess, snapshot_path)   
    train_writer.close()    

    print('shut down queue')
    train_enqueue_thread.join()
    valid_enqueue_thread.join()

    sess.run(
      [train_queue.close(cancel_pending_enqueues=True),
       valid_queue.close(cancel_pending_enqueues=True)])
    coord.request_stop()
    coord.join(threads)   
  
if __name__ == '__main__':
  tf.app.run(main=main)
