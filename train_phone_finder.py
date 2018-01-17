import os
import math
import time
import tensorflow as tf 
import matplotlib.pyplot as plt 
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import numpy as np
import cv2
import glob

layer_norm = tf.contrib.layers.layer_norm

def conv2d(x, W, b, strides = 1):
    x = tf.nn.conv2d(x,W, strides=[1,strides,strides,1], padding='SAME')
    x = tf.nn.bias_add(x,b)
    return(tf.nn.relu(x))

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')

def fclayer(x, W, b):
    fc = tf.sigmoid(tf.add( tf.matmul(x,W), b))
    return fc

def init_w(shape):
    return tf.Variable(tf.truncated_normal(shape,mean=0.0, stddev = 0.1))

def init_b(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def conv_layer(input, y, k=5, f=3, do_layer_norm=True):
    w = init_w([k,k,3,f])
    b = init_b([f])
    y = tf.reshape(y, [-1,2])
    conv = conv2d(input, w, b)
    maxp = maxpool2d(conv)
    if do_layer_norm:
        maxp = layer_norm(maxp)
    return maxp

### Parameters
totaltime=0
filename = 'data/labels.txt'
filenames = []
labels = []
input_image = []
lbl = []
lambda_l1 = 0.0000005
lambda_l2 = 0.0000005

#reading file and extracting path and labels
with open(filename, 'r') as File:
    infoFile = File.readlines()     
    for line in infoFile: 
        words = line.split(' ')
        filenames.append('data/raw_images/'+words[0])
        labels.append(float(words[1]) )
        labels.append(float(words[2]) )
NumFiles = len(filenames)
tfilenames = ops.convert_to_tensor(filenames, dtype = dtypes.string)
tlabels = ops.convert_to_tensor(labels, dtype=dtypes.float32)
filename_queue = tf.train.slice_input_producer([tfilenames, tlabels], num_epochs=10, shuffle=False, capacity = NumFiles)
rawIm = tf.read_file(filename_queue[0])
decodedIm = tf.image.decode_jpeg(rawIm)
label_queue = filename_queue[1]

### Build Graph
x = tf.placeholder(tf.float32, shape=[None,479220])
y = tf.placeholder(tf.float32, shape=[None,2])
x_image = tf.reshape(x, [-1,326,490,3])

layer1 = conv_layer(x_image, y)
layer2 = conv_layer(layer1, y)
layer3 = conv_layer(layer2, y)
layer4 = conv_layer(layer3, y)
layer5 = conv_layer(layer4, y)
layer6 = conv_layer(layer5, y)

conv_flatten = tf.reshape(layer6,[-1,144])
w_fc = init_w([144,2])
b_fc = init_b([2])
fc = fclayer(conv_flatten,w_fc, b_fc)

### Loss and Regularization
loss = tf.reduce_sum(tf.square(fc - y))
"""
# L1
l1_regularizer = tf.contrib.layers.l1_regularizer(scale=lambda_l1, scope=None)
penalty = tf.contrib.layers.apply_regularization(l1_regularizer, tf.trainable_variables())
# L2
for w in tf.trainable_variables():
    if penalty:
        penalty += lambda_l2 * tf.nn.l2_loss(w)
    else:
        penalty = lambda_l2 * tf.nn.l2_loss(w)
loss += penalty
"""

### Save Graph
saver = tf.train.Saver()

### Set Optimizer
#train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
train_step = tf.train.AdadeltaOptimizer(learning_rate=0.1).minimize(loss)
#train_step = tf.train.AdamOptimizer().minimize(loss)

### Initialize Graph
init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
sess = tf.InteractiveSession()
sess.run(init_op)

### Initialize Parameters
label_value, Tolerance, no_epoch, label_counter, Train_Checker, model_id, loss_to_be_minimized = [], 0, 0, 0, [], 0, 0

### Start Training
with sess.as_default():	
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord = coord)
    writer = tf.summary.FileWriter('results/graphs', sess.graph)
    flag, lbl_array, img_array = 0, [], [] 
	
    while(True):
        flag = flag + 1
        begtime = time.time()
        i = 0
        Train_Checker.append(loss_to_be_minimized)
        loss_to_be_minimized = 0
        for i in range(NumFiles):		
            if flag<=1:
                nm, image, lb = sess.run([filename_queue[0], decodedIm, label_queue])					
                labels =np.reshape(labels, (-1,2))
                lbl = labels[i]
                lbl_array.append(lbl)
                input_image = (sess.run(tf.reshape(image, [479220])))
                img_array.append(input_image)
			
            no_of_times_run = 0
            while(True):
                train_step.run(feed_dict={x:[img_array[i]], y:[lbl_array[i]]})
                no_of_times_run = no_of_times_run + 1
                if no_of_times_run>3:
                    break
           
            tmp_loss = sess.run(loss, feed_dict={x:[img_array[i]], y:[lbl_array[i]]})
            loss_to_be_minimized += tmp_loss

        endtime = time.time()
        totaltime = totaltime + (endtime-begtime)
        model_str = 'results/models/model_' + str(model_id) + '.ckpt'
        save_path = saver.save(sess, model_str)	
        model_id += 1
        loss_to_be_minimized = math.sqrt(loss_to_be_minimized / NumFiles)

        print ("Epoch: "+ str(flag)+ "\t"+ "Total Error: "+str(loss_to_be_minimized)+ "\t"+ "Tolerance: "+ str(Tolerance)+ "\t" + "Time Taken: "+ str(endtime- begtime))

        Train_Checker.append(loss_to_be_minimized) 
        if (loss_to_be_minimized < 0.000001):
            break
        if (Train_Checker[0] <= Train_Checker[1] and flag > 1):
            Tolerance = Tolerance + 1
            if (Tolerance > 30):	
                break
        del Train_Checker[:]
    
    coord.request_stop()
    coord.join(threads)
    writer.close()
	
print("TotalTime Taken: " + str(totaltime))
