# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 23:20:34 2018

@author: Shiyao Han
"""

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
# (number 0 - 9)
y = tf.placeholder(tf.float32, [None, 10])  
# keep_prob% is working, e.g. 1--all are working
keep_prob = tf.placeholder(tf.float32)

# creat 1st layer, keep_prob: 
W1 = tf.Variable(tf.truncated_normal([784,200], stddev=0.1))
print(type(W1))
b1 = tf.Variable(tf.zeros([200])+0.1)
L1 = tf.nn.tanh(tf.matmul(x,W1)+b1)
L1_drop = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.truncated_normal([200,200], stddev=0.1))
b2 = tf.Variable(tf.zeros([200])+0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop,W2)+b2)
L2_drop = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.truncated_normal([200,100], stddev=0.1))
b3 = tf.Variable(tf.zeros([100])+0.1)
L3 = tf.nn.tanh(tf.matmul(L2_drop,W3)+b3)
L3_drop = tf.nn.dropout(L3, keep_prob)

W4 = tf.Variable(tf.truncated_normal([100,10], stddev=0.1))
b4 = tf.Variable(tf.zeros([10])+0.1)

prediction = tf.nn.softmax(tf.matmul(L3_drop, W4)+b4)



# quadratic cost function
#loss = tf.reduce_mean(tf.square(y-prediction))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = prediction))
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
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys, keep_prob: 1.0})
        test_acc  = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob: 1.0})
        train_acc = sess.run(accuracy, feed_dict={x:mnist.train.images, y:mnist.train.labels, keep_prob: 1.0})
        
        
        print("Iter" + str(epoch) + ", train accuracy" + str(train_acc))
        print("Iter" + str(epoch) + ", test accuracy" + str(test_acc))
            