# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 14:03:29 2018

@author: Shiyao Han
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# input data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 
# width of input 
n_inputs = 28
# height of image
max_time = 28
# hidden neural 
lstm_size = 100
n_classes = 10
batch_size = 50
n_batch = mnist.train.num_examples // batch_size

# initialize weight
## in fully connected neutral net(before), the 'shape' is the whole image shape, 
## but here, since we want to do the local convolution, the shape is flexible
def weight_variable(shape):
    initial = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    return initial
    
# initialize bias
## in fully connected neutral net(before), the 'shape' is the whole image shape, 
## but here, since we want to do the local convolution, the shape is flexible
def bias_variable(shape):
    initial = tf.Variable(tf.constant(0.1, shape=shape))
    return initial

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

 
# define 2 placeholders for data(image) and label
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])  


# initialize weight and bias
weight = tf.Variable(tf.truncated_normal([lstm_size, n_classes], stddev=0.1))
bias = tf.Variable(tf.constant(0.1, shape = [n_classes]))

def RNN(X, weight, bias):
    
    inputs = tf.reshape(X, [-1, max_time, n_inputs])
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)
    results = tf.nn.softmax(tf.matmul(final_state[1], weight) + bias)
    return results


prediction = RNN(x, weight, bias)

## loss
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = 7, logits = prediction))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = prediction))
## gradient descent 
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# if equal- true, not - false
correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
# if prediction is correct, true, correct_prediction = 1, 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))




with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(6):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys})
        
        test_acc  = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
        print("Iter" + str(epoch) + ", test accuracy" + str(test_acc))