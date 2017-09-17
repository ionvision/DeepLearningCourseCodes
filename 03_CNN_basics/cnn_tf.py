
# coding: utf-8

# # TensorFlow卷积神经网络(CNN)示例 - 高级API
# ### Convolutional Neural Network Example - tf.layers API

# ## CNN网络结构图示
# 
# ![CNN](http://personal.ie.cuhk.edu.hk/~ccloy/project_target_code/images/fig3.png)
# 
# ## MNIST数据库
# 
# ![MNIST Dataset](http://neuralnetworksanddeeplearning.com/images/mnist_100_digits.png)
# 
# More info: http://yann.lecun.com/exdb/mnist/

# In[1]:

from __future__ import division, print_function, absolute_import

# Import MNIST data，MNIST数据集导入
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


# In[2]:

# Training Parameters，超参数
learning_rate = 0.001 #学习率
num_steps = 2000 # 训练步数
batch_size = 128 # 训练数据批的大小

# Network Parameters，网络参数
num_input = 784 # MNIST数据输入 (img shape: 28*28)
num_classes = 10 # MNIST所有类别 (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units，保留神经元相应的概率


# In[3]:

# Create the neural network，创建深度神经网络
def conv_net(x_dict, n_classes, dropout, reuse, is_training):
    
    # Define a scope for reusing the variables，确定命名空间
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator类型的输入为像素
        x = x_dict['images']

        # MNIST数据输入格式为一位向量，包含784个特征 (28*28像素)
        # 用reshape函数改变形状以匹配图像的尺寸 [高 x 宽 x 通道数]
        # 输入张量的尺度为四维: [(每一)批数据的数目, 高，宽，通道数]
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        # 卷积层，32个卷积核，尺寸为5x5
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        # 最大池化层，步长为2，无需学习任何参量
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # 卷积层，32个卷积核，尺寸为5x5
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        # 最大池化层，步长为2，无需学习任何参量
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # 展开特征为一维向量，以输入全连接层
        fc1 = tf.contrib.layers.flatten(conv2)

        # 全连接层
        fc1 = tf.layers.dense(fc1, 1024)
        # 应用Dropout (训练时打开，测试时关闭)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # 输出层，预测类别
        out = tf.layers.dense(fc1, n_classes)

    return out


# In[4]:

# 确定模型功能 (参照TF Estimator模版)
def model_fn(features, labels, mode):
    
    # 构建神经网络
    # 因为dropout在训练与测试时的特性不一，我们此处为训练和测试过程创建两个独立但共享权值的计算图
    logits_train = conv_net(features, num_classes, dropout, reuse=False, is_training=True)
    logits_test = conv_net(features, num_classes, dropout, reuse=True, is_training=False)
    
    # 预测
    pred_classes = tf.argmax(logits_test, axis=1)
    pred_probas = tf.nn.softmax(logits_test)
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes) 
        
    # 确定误差函数与优化器
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())
    
    # 评估模型精确度
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
    
    # TF Estimators需要返回EstimatorSpec
    estim_specs = tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=pred_classes,
      loss=loss_op,
      train_op=train_op,
      eval_metric_ops={'accuracy': acc_op})

    return estim_specs


# In[5]:

# 构建Estimator
model = tf.estimator.Estimator(model_fn)


# In[6]:

# 确定训练输入函数
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.train.images}, y=mnist.train.labels,
    batch_size=batch_size, num_epochs=None, shuffle=True)
# 开始训练模型
model.train(input_fn, steps=num_steps)


# In[7]:

# 评判模型
# 确定评判用输入函数
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.test.images}, y=mnist.test.labels,
    batch_size=batch_size, shuffle=False)
model.evaluate(input_fn)


# In[8]:

# 预测单个图像
n_images = 4
# 从数据集得到测试图像
test_images = mnist.test.images[:n_images]
# 准备输入数据
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': test_images}, shuffle=False)
# 用训练好的模型预测图片类别
preds = list(model.predict(input_fn))

# 可视化显示
for i in range(n_images):
    plt.imshow(np.reshape(test_images[i], [28, 28]), cmap='gray')
    plt.show()
    print("Model prediction:", preds[i])

