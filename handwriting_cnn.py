# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 15:48:12 2018

@author: ga47soz
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# input data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# size of batch(instead of dealing a single sample, it processes a batch of samples)
batch_size = 100
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
# (number 0 - 9)
y = tf.placeholder(tf.float32, [None, 10])  


# change the format of x to 4D [batch, in_height, in_width, in_channels]
x_image = tf.reshape(x,[-1,28,28,1])

#initialize the weight and bias of the first conv
W_conv1 = weight_variable([5,5,1,32]) # 5*5windiw, 32 conv kernel on one chaneel
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#initialize the weight and bias of the first conv
W_conv2 = weight_variable([5,5,32,64]) # 5*5windiw, 64 conv kernel on 32 chaneel
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


#initialize full connected layers
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)

# keep_prob% is working, e.g. 1--all are working
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)



W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])


prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2)+b_fc2)
## loss
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = prediction))
## gradient descent 
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# if equal- true, not - false
correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
# if prediction is correct, true, correct_prediction = 1, 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys, keep_prob: 0.7})
        
        test_acc  = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob: 1.0})
        print("Iter" + str(epoch) + ", test accuracy" + str(test_acc))