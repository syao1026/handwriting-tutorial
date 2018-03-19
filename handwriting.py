# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 20:14:42 2018

@author: Shiyao Han
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# input data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# size of batch(instead of dealing a single sample, it processes a batch of samples)
batch_size = 100
n_batch = mnist.train.num_examples // batch_size

# define 2 placeholders for data(image) and label
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])  #(number 0 - 9)

# creat simpel network
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x,W)+b)

# quadratic cost function
loss = tf.reduce_mean(tf.square(y-prediction))
# gradient descent 
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init = tf.global_variables_initializer()

# if equal- true, not - false
correct_prediction = tf.equal(tf.arg_max(y,1), tf.arg_max(prediction,1))

# if prediction is correct, true, correct_prediction = 1, 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            # batch_xs: batch_size samples batch_ys: batch_size labels
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # put sampels for train
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys})
        acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
        print("Iter" + str(epoch) + ", test accuracy" + str(acc))
            