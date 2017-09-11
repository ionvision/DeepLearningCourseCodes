
# coding: utf-8

# # 神经网络，TensorFlow示例
# 
# 两层(隐含层)全连接神经网络

# ## 神经网络结构概览
# 
# <img src="http://cs231n.github.io/assets/nn1/neural_net2.jpeg" alt="nn" style="width: 400px;"/>
# 
# ## MNIST 数据库概览
# 
# MNIST手写数字数据集. 共有60,000张训练图片，10,000张测试图片. 所有图片已做过归一化与中心化，尺寸统一为28x28，像素值范围从0到1. 为了简单期间，输入时每张图片不以28x28的矩阵形式输入，而是按列排列，转为一维的长度为784=28x28的特征，输入神经网络。
# 
# ![MNIST 数据集](http://neuralnetworksanddeeplearning.com/images/mnist_100_digits.png)
# 
# 更多信息: http://yann.lecun.com/exdb/mnist/

# In[1]:

from __future__ import print_function

# Import MNIST data，准备MNIST输入数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf


# In[2]:

# Hyper Parameters，超参
learning_rate = 0.1
num_steps = 500
batch_size = 128
display_step = 100

# Network Parameters，网络参数
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input，TensorFlow图模型结构的输入定义
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])


# In[3]:

# Store layers weight & bias，定义需要学习的网络参数，即网络权值
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}


# In[4]:

# Create model，定义神经网络基本结构
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


# In[5]:

# Construct model，初始化所定义神经网络结构的实例
logits = neural_net(X)

# Define loss and optimizer，定义误差函数与优化方式
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)，定义检验模型效果的方式
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)，初始化所有定义的网络变量
init = tf.global_variables_initializer()


# In[6]:

# Start training，开启session，将所有定义编译成实际的TensorFlow图模型并运行
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)，进行前向传播，后向传播，以及优化
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy，计算每一批数据的误差及准确度
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " +                   "{:.4f}".format(loss) + ", Training Accuracy= " +                   "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for MNIST test images，计算测试数据上的准确度
    print("Testing Accuracy:",         sess.run(accuracy, feed_dict={X: mnist.test.images,
                                      Y: mnist.test.labels}))

