
# coding: utf-8

# # MNIST 数据集 简介
# 作为最经典的深度学习数据集，MNISIT包含65,000个灰度书写数字图片，尺寸均为`28x28`，其中55,000个用于训练，10,000个用于测试
# 所有图片已归一化与中心化，像素值从0到1。

# ## 使用
# 我们使用TensorFlow的input_data函数进行数据准备与输入，它能帮助：
# * 自动下载数据
# * 将数据集load成numpy array的形式

# In[1]:

from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/", one_hot=True)

# Load data
x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels


# ## 数据维度
# 每张图片维度为784 (28x28x1)

# In[2]:

print("x_train: ", x_train.shape)
print("y_train: ", y_train.shape)
print("x_test: ", x_test.shape)
print("y_test: ", y_test.shape)


# ## 数据可视化 
# 使用matplotlib可视化MNIST:

# In[3]:

import matplotlib.pyplot as plt
import numpy as np

def plot_mnist(data, classes):
    
    for i in range(10):
        idxs = (classes == i)
        
        # get 10 images for class i
        images = data[idxs][0:10]
            
        for j in range(5):   
            plt.subplot(5, 10, i + j*10 + 1)
            plt.imshow(images[j].reshape(28, 28), cmap='gray')
            # print a title only once for each class
            if j == 0:
                plt.title(i)
            plt.axis('off')
    plt.show()

classes = np.argmax(y_train, 1)
plot_mnist(x_train, classes)

