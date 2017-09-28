
# coding: utf-8

# # 卷积神经网络示例与各层可视化

# In[1]:

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
get_ipython().magic(u'matplotlib inline')
print ("当前TensorFlow版本为 [%s]" % (tf.__version__))
print ("所有包载入完毕")


# ## 载入 MNIST

# In[2]:

mnist = input_data.read_data_sets('data/', one_hot=True)
trainimg   = mnist.train.images
trainlabel = mnist.train.labels
testimg    = mnist.test.images
testlabel  = mnist.test.labels
print ("MNIST ready")


# ## 定义模型

# In[3]:

# NETWORK TOPOLOGIES
n_input    = 784
n_channel  = 64 
n_classes  = 10  

# INPUTS AND OUTPUTS
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
    
# NETWORK PARAMETERS
stddev = 0.1
weights = {
    'c1': tf.Variable(tf.random_normal([7, 7, 1, n_channel], stddev=stddev)),
    'd1': tf.Variable(tf.random_normal([14*14*64, n_classes], stddev=stddev))
}
biases = {
    'c1': tf.Variable(tf.random_normal([n_channel], stddev=stddev)),
    'd1': tf.Variable(tf.random_normal([n_classes], stddev=stddev))
}
print ("NETWORK READY")


# ## 定义图结构

# In[4]:

# MODEL
def CNN(_x, _w, _b):
    # RESHAPE
    _x_r = tf.reshape(_x, shape=[-1, 28, 28, 1])
    # CONVOLUTION
    _conv1 = tf.nn.conv2d(_x_r, _w['c1'], strides=[1, 1, 1, 1], padding='SAME')
    # ADD BIAS
    _conv2 = tf.nn.bias_add(_conv1, _b['c1'])
    # RELU
    _conv3 = tf.nn.relu(_conv2)
    # MAX-POOL
    _pool  = tf.nn.max_pool(_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # VECTORIZE
    _dense = tf.reshape(_pool, [-1, _w['d1'].get_shape().as_list()[0]])
    # DENSE
    _logit = tf.add(tf.matmul(_dense, _w['d1']), _b['d1'])
    _out = {
        'x_r': _x_r, 'conv1': _conv1, 'conv2': _conv2, 'conv3': _conv3
        , 'pool': _pool, 'dense': _dense, 'logit': _logit
    }
    return _out

# PREDICTION
cnnout = CNN(x, weights, biases)

# LOSS AND OPTIMIZER
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=y, logits=cnnout['logit']))
optm = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost) 
corr = tf.equal(tf.argmax(cnnout['logit'], 1), tf.argmax(y, 1))    
accr = tf.reduce_mean(tf.cast(corr, "float"))

# INITIALIZER
init = tf.global_variables_initializer()
print ("FUNCTIONS READY")


# ## 存储

# In[5]:

savedir = "nets/cnn_mnist_simple/"
saver = tf.train.Saver(max_to_keep=3)
save_step = 4
if not os.path.exists(savedir):
    os.makedirs(savedir)
print ("SAVER READY")


# ## 运行

# In[6]:

# PARAMETERS
training_epochs = 20
batch_size      = 100
display_step    = 4
# LAUNCH THE GRAPH
sess = tf.Session()
sess.run(init)
# OPTIMIZE
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(mnist.train.num_examples/batch_size)
    # ITERATION
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feeds = {x: batch_xs, y: batch_ys}
        sess.run(optm, feed_dict=feeds)
        avg_cost += sess.run(cost, feed_dict=feeds)
    avg_cost = avg_cost / total_batch
    # DISPLAY
    if (epoch+1) % display_step == 0:
        print ("Epoch: %03d/%03d cost: %.9f" % (epoch+1, training_epochs, avg_cost))
        feeds = {x: batch_xs, y: batch_ys}
        train_acc = sess.run(accr, feed_dict=feeds)
        print ("TRAIN ACCURACY: %.3f" % (train_acc))
        feeds = {x: mnist.test.images, y: mnist.test.labels}
        test_acc = sess.run(accr, feed_dict=feeds)
        print ("TEST ACCURACY: %.3f" % (test_acc))
    # SAVE
    if (epoch+1) % save_step == 0:
        savename = savedir+"net-"+str(epoch+1)+".ckpt"
        saver.save(sess, savename)
        print ("[%s] SAVED." % (savename))
print ("OPTIMIZATION FINISHED")


# ## 恢复

# In[7]:

do_restore = 0
if do_restore == 1:
    sess = tf.Session()
    epoch = 20
    savename = savedir+"net-"+str(epoch)+".ckpt"
    saver.restore(sess, savename)
    print ("NETWORK RESTORED")
else:
    print ("DO NOTHING")


# ## CNN如何工作

# In[8]:

input_r = sess.run(cnnout['x_r'], feed_dict={x: trainimg[0:1, :]})
conv1   = sess.run(cnnout['conv1'], feed_dict={x: trainimg[0:1, :]})
conv2   = sess.run(cnnout['conv2'], feed_dict={x: trainimg[0:1, :]})
conv3   = sess.run(cnnout['conv3'], feed_dict={x: trainimg[0:1, :]})
pool    = sess.run(cnnout['pool'], feed_dict={x: trainimg[0:1, :]})
dense   = sess.run(cnnout['dense'], feed_dict={x: trainimg[0:1, :]})
out     = sess.run(cnnout['logit'], feed_dict={x: trainimg[0:1, :]}) 


# ## 输入

# In[9]:

print ("Size of 'input_r' is %s" % (input_r.shape,))
label = np.argmax(trainlabel[0, :])
print ("Label is %d" % (label))

# PLOT
plt.matshow(input_r[0, :, :, 0], cmap=plt.get_cmap('gray'))
plt.title("Label of this image is " + str(label) + "")
plt.colorbar()
plt.show()


# # CONV 卷积层

# In[10]:

print ("SIZE OF 'CONV1' IS %s" % (conv1.shape,))
for i in range(3):
    plt.matshow(conv1[0, :, :, i], cmap=plt.get_cmap('gray'))
    plt.title(str(i) + "th conv1")
    plt.colorbar()
    plt.show()


# ## CONV + BIAS

# In[11]:

print ("SIZE OF 'CONV2' IS %s" % (conv2.shape,))
for i in range(3):
    plt.matshow(conv2[0, :, :, i], cmap=plt.get_cmap('gray'))
    plt.title(str(i) + "th conv2")
    plt.colorbar()
    plt.show()


# ## CONV + BIAS + RELU

# In[12]:

print ("SIZE OF 'CONV3' IS %s" % (conv3.shape,))
for i in range(3):
    plt.matshow(conv3[0, :, :, i], cmap=plt.get_cmap('gray'))
    plt.title(str(i) + "th conv3")
    plt.colorbar()
    plt.show()


# ## POOL

# In[13]:

print ("SIZE OF 'POOL' IS %s" % (pool.shape,))
for i in range(3):
    plt.matshow(pool[0, :, :, i], cmap=plt.get_cmap('gray'))
    plt.title(str(i) + "th pool")
    plt.colorbar()
    plt.show()


# ## DENSE

# In[14]:

print ("SIZE OF 'DENSE' IS %s" % (dense.shape,))
print ("SIZE OF 'OUT' IS %s" % (out.shape,))
plt.matshow(out, cmap=plt.get_cmap('gray'))
plt.title("OUT")
plt.colorbar()
plt.show()


# ## CONVOLUTION FILTER 卷积核

# In[15]:

wc1 = sess.run(weights['c1'])
print ("SIZE OF 'WC1' IS %s" % (wc1.shape,))
for i in range(3):
    plt.matshow(wc1[:, :, 0, i], cmap=plt.get_cmap('gray'))
    plt.title(str(i) + "th conv filter")
    plt.colorbar()
    plt.show()

