
# coding: utf-8

# # TensorFlow卷积神经网络(CNN)示例 - 原生API
# ### Convolutional Neural Network Example - Raw API
# 
# 

# ## CNN网络结构图示
# 
# ![CNN](http://personal.ie.cuhk.edu.hk/~ccloy/project_target_code/images/fig3.png)
# 
# ## MNIST数据集
# 
# 
# ![MNIST Dataset](http://neuralnetworksanddeeplearning.com/images/mnist_100_digits.png)
# 
# More info: http://yann.lecun.com/exdb/mnist/

# In[1]:

from __future__ import division, print_function, absolute_import

import tensorflow as tf

# Import MNIST data，MNIST数据集导入
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


# In[2]:

# Hyper-parameters，超参数
learning_rate = 0.001
num_steps = 500
batch_size = 128
display_step = 10

# Network Parameters，网络参数
num_input = 784 # MNIST数据输入 (img shape: 28*28)
num_classes = 10 # MNIST所有类别 (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units，保留神经元相应的概率

# tf Graph input，TensorFlow图结构输入
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)，保留i


# In[3]:

# Create some wrappers for simplicity，创建基础卷积函数，简化写法
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation，卷积层，包含bias与非线性relu激励
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper，最大池化层
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model，创建模型
def conv_net(x, weights, biases, dropout):
    # MNIST数据为维度为1，长度为784 (28*28 像素)的
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer，卷积层
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)，最大池化层／下采样
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer，卷积层
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)，最大池化层／下采样
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer，全连接网络
    # Reshape conv2 output to fit fully connected layer input，调整conv2层输出的结果以符合全连接层的需求
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout，应用dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction，最后输出预测
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


# In[4]:

# Store layers weight & bias 存储每一层的权值和全差
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Construct model，构建模型
logits = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer，定义误差函数与优化器
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


# Evaluate model，评估模型
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)，初始化图结构所有变量
init = tf.global_variables_initializer()


# In[5]:

# Start training，开始训练
with tf.Session() as sess:

    # Run the initializer，初始化
    sess.run(init)

    for step in range(1, num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)，优化
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y,
                                                                 keep_prob: 1.0})
            print("Step " + str(step) + ", Minibatch Loss= " +                   "{:.4f}".format(loss) + ", Training Accuracy= " +                   "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 256 MNIST test images，以每256个测试图像为例，
    print("Testing Accuracy:",         sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
                                      Y: mnist.test.labels[:256],
                                      keep_prob: 1.0}))

