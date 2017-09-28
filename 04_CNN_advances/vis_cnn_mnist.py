
# coding: utf-8

# # 基于TensorBoard的CNN可视化

# In[1]:

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
get_ipython().magic(u'matplotlib inline')
mnist = input_data.read_data_sets('data/', one_hot=True)
trainimg   = mnist.train.images
trainlabel = mnist.train.labels
testimg    = mnist.test.images
testlabel  = mnist.test.labels
print ("Packages loaded.")


# # 定义网络
# ## tf.variable_scope("NAME"):

# In[2]:

# Parameters
learning_rate   = 0.001
training_epochs = 5
batch_size      = 100
display_step    = 1

# Network
n_input  = 784
n_output = 10
with tf.variable_scope("CNN_WEIGHTS"):
    weights  = {
        'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=0.1)),
        'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.1)),
        'wd1': tf.Variable(tf.random_normal([7*7*128, 1024], stddev=0.1)),
        'wd2': tf.Variable(tf.random_normal([1024, n_output], stddev=0.1))
    }
with tf.variable_scope("CNN_BIASES"):
    biases   = {
        'bc1': tf.Variable(tf.random_normal([64], stddev=0.1)),
        'bc2': tf.Variable(tf.random_normal([128], stddev=0.1)),
        'bd1': tf.Variable(tf.random_normal([1024], stddev=0.1)),
        'bd2': tf.Variable(tf.random_normal([n_output], stddev=0.1))
    }


# In[3]:

def conv_basic(_input, _w, _b, _keepratio):
    # Input
    with tf.variable_scope("INPUT_LAYER"):
        _input_r = tf.reshape(_input, shape=[-1, 28, 28, 1])
    # Conv1
    with tf.variable_scope("CNN_CONV_1"):
        _conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(_input_r, _w['wc1']
                    , strides=[1, 1, 1, 1], padding='SAME'), _b['bc1']))
    with tf.variable_scope("CNN_POOL_1"):
        _pool1 = tf.nn.max_pool(_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1]
                                , padding='SAME')
        _pool_dr1 = tf.nn.dropout(_pool1, _keepratio)
    # Conv2
    with tf.variable_scope("CNN_CONV_2"):
        _conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(_pool_dr1, _w['wc2']
                    , strides=[1, 1, 1, 1], padding='SAME'), _b['bc2']))
    with tf.variable_scope("CNN_POOL_2"):
        _pool2 = tf.nn.max_pool(_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1]
                                , padding='SAME')
        _pool_dr2 = tf.nn.dropout(_pool2, _keepratio)
    with tf.variable_scope("FC_1"):
        # Vectorize
        _dense1 = tf.reshape(_pool_dr2, [-1, _w['wd1'].get_shape().as_list()[0]])
        # Fc1
        _fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(_dense1, _w['wd1']), _b['bd1']))
        _fc_dr1 = tf.nn.dropout(_fc1, _keepratio)
    with tf.variable_scope("FC_2"):
        # Fc2
        _out = tf.add(tf.matmul(_fc_dr1, _w['wd2']), _b['bd2'])
    # Return everything
    out = {
        'input_r': _input_r, 'conv1': _conv1, 'pool1': _pool1, 'pool1_dr1': _pool_dr1,
        'conv2': _conv2, 'pool2': _pool2, 'pool_dr2': _pool_dr2, 'dense1': _dense1,
        'fc1': _fc1, 'fc_dr1': _fc_dr1, 'out': _out }
    return out

print ("Network ready")


# # 定义模型

# In[4]:

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input], name="CNN_INPUT_x")
y = tf.placeholder(tf.float32, [None, n_output], name="CNN_TARGET_y")
keepratio = tf.placeholder(tf.float32, name="CNN_DROPOUT_keepratio")

# Functions! 
pred = conv_basic(x, weights, biases, keepratio)['out']
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optm = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
corr = tf.equal(tf.argmax(pred,1), tf.argmax(y,1)) # Count corrects
accr = tf.reduce_mean(tf.cast(corr, tf.float32)) # Accuracy
init = tf.initialize_all_variables()
print ("Functions ready")


# ## 准备Summary

# In[8]:

# Do some optimizations
sess = tf.Session()
sess.run(init)

# Summary writer
tf.summary.scalar('cross entropy', cost)
tf.summary.scalar('accuracy', accr)
merged = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('/tmp/tf_logs/cnn_mnist'
                                        , graph=sess.graph)
print ("Summary ready")


# # 运行

# In[ ]:

print ("Start!")
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(mnist.train.num_examples/batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # Fit training using batch data
        summary, _ = sess.run([merged, optm]
                        , feed_dict={x: batch_xs, y: batch_ys, keepratio:0.7})
        # Compute average loss
        avg_cost += sess.run(cost
                , feed_dict={x: batch_xs, y: batch_ys, keepratio:1.})/total_batch
        # Add summary
        summary_writer.add_summary(summary, epoch*total_batch+i)
    # Display logs per epoch step
    if epoch % display_step == 0: 
        print ("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
        train_acc = sess.run(accr, feed_dict={x: batch_xs, y: batch_ys, keepratio:1.})
        print (" Training accuracy: %.3f" % (train_acc))
        test_acc = sess.run(accr, feed_dict={x: testimg, y: testlabel, keepratio:1.})
        print (" Test accuracy: %.3f" % (test_acc))

print ("Optimization Finished.")


# # 在终端或命令行中运行下面的命令
# ### `tensorboard --logdir=/tmp/tf_logs/cnn_mnist`
# # 在浏览器中打开下列地址，查看TensorBoard可视化结果
# ### `http://localhost:6006/`

# <img src="images/tsboard/cnn_mnist.png">
