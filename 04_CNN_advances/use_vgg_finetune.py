
# coding: utf-8

# # 使用预训练的VGG模型Fine-tune CNN

# In[1]:

# Import packs
import numpy as np
import os
import scipy.io
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import skimage.io
import skimage.transform
import tensorflow as tf
get_ipython().magic(u'matplotlib inline')
cwd = os.getcwd()
print ("Package loaded")
print ("Current folder is %s" % (cwd) )


# In[2]:

# 下载预先训练好的vgg-19模型，为Matlab的.mat格式，之后会用scipy读取
# (注意此版本模型与此处http://www.vlfeat.org/matconvnet/pretrained/最新版本不同)
import os.path
if not os.path.isfile('./data/imagenet-vgg-verydeep-19.mat'):
    get_ipython().system(u'wget -O data/imagenet-vgg-verydeep-19.mat http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat')


# # 载入图像，调节尺寸，生成数据集

# In[3]:

# Configure the locations of the images and reshaping sizes
# ------------------------------------------------------------------- #
paths = {"images/cats", "images/dogs"}
imgsize = [64, 64]      # The reshape size
use_gray = 0            # Grayscale
data_name = "data4vgg"  # Save name
valid_exts = [".jpg",".gif",".png",".tga", ".jpeg"]
# ------------------------------------------------------------------- #

imgcnt = 0
nclass = len(paths)
for relpath in paths:
    fullpath = cwd + "/" + relpath
    flist = os.listdir(fullpath)
    for f in flist:
        if os.path.splitext(f)[1].lower() not in valid_exts:
            continue
        fullpath = os.path.join(fullpath, f)
        imgcnt = imgcnt + 1
# Grayscale
def rgb2gray(rgb):
    if len(rgb.shape) is 3:
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    else:
        print ("Current Image is GRAY!")
        return rgb
if use_gray:
    totalimg   = np.ndarray((imgcnt, imgsize[0]*imgsize[1]))
else:
    totalimg   = np.ndarray((imgcnt, imgsize[0]*imgsize[1]*3))
totallabel = np.ndarray((imgcnt, nclass))
imgcnt     = 0
for i, relpath in zip(range(nclass), paths):
    path = cwd + "/" + relpath
    flist = os.listdir(path)
    for f in flist:
        if os.path.splitext(f)[1].lower() not in valid_exts:
            continue
        fullpath = os.path.join(path, f)
        currimg  = imread(fullpath)
        # Convert to grayscale  
        if use_gray:
            grayimg  = rgb2gray(currimg)
        else:
            grayimg  = currimg
        # Reshape
        graysmall = imresize(grayimg, [imgsize[0], imgsize[1]])/255.
        grayvec   = np.reshape(graysmall, (1, -1))
        # Save 
        totalimg[imgcnt, :] = grayvec
        totallabel[imgcnt, :] = np.eye(nclass, nclass)[i]
        imgcnt    = imgcnt + 1
        
# Divide total data into training and test set
randidx    = np.random.randint(imgcnt, size=imgcnt)
trainidx   = randidx[0:int(4*imgcnt/5)]
testidx    = randidx[int(4*imgcnt/5):imgcnt]
trainimg   = totalimg[trainidx, :]
trainlabel = totallabel[trainidx, :]
testimg    = totalimg[testidx, :]
testlabel  = totallabel[testidx, :]
ntrain     = trainimg.shape[0]
nclass     = trainlabel.shape[1]
dim        = trainimg.shape[1]
ntest      = testimg.shape[0]

print ("Number of total images is %d (train: %d, test: %d)" 
       % (imgcnt, ntrain, ntest)) 
print ("Shape of an image is (%d, %d, %d)" % (imgsize[0], imgsize[1], 3))


# # 定义VGG网络结构

# In[4]:

def net(data_path, input_image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )
    data = scipy.io.loadmat(data_path)
    mean = data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    weights = data['layers'][0]
    net = {}
    current = input_image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            current = _conv_layer(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current)
        elif kind == 'pool':
            current = _pool_layer(current)
        net[name] = current
    assert len(net) == len(layers)
    return net, mean_pixel
def _conv_layer(input, weights, bias):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1),
            padding='SAME')
    return tf.nn.bias_add(conv, bias)
def _pool_layer(input):
    return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
            padding='SAME')
def preprocess(image, mean_pixel):
    return image - mean_pixel
def unprocess(image, mean_pixel):
    return image + mean_pixel
print ("VGG net ready")


# # 使用VGG计算卷积特征图

# In[5]:

# Preprocess
trainimg_tensor = np.ndarray((ntrain, imgsize[0], imgsize[1], 3))
testimg_tensor = np.ndarray((ntest, imgsize[0], imgsize[1], 3))
for i in range(ntrain):
    currimg = trainimg[i, :]
    currimg = np.reshape(currimg, [imgsize[0], imgsize[1], 3])
    trainimg_tensor[i, :, :, :] = currimg 
print ("Shape of trainimg_tensor is %s" % (trainimg_tensor.shape,))    

for i in range(ntest):
    currimg = testimg[i, :]
    currimg = np.reshape(currimg, [imgsize[0], imgsize[1], 3])
    testimg_tensor[i, :, :, :] = currimg 
print ("Shape of trainimg_tensor is %s" % (testimg_tensor.shape,))
    
# Get conv features
VGG_PATH = cwd + "/data/imagenet-vgg-verydeep-19.mat"
with tf.Graph().as_default(), tf.Session() as sess:
    with tf.device("/cpu:0"):
        img_placeholder = tf.placeholder(tf.float32
                                         , shape=(None, imgsize[0], imgsize[1], 3))
        nets, mean_pixel = net(VGG_PATH, img_placeholder)
        train_features = nets['relu5_4'].eval(feed_dict={img_placeholder: trainimg_tensor})
        test_features  = nets['relu5_4'].eval(feed_dict={img_placeholder: testimg_tensor})
print("Convolutional map extraction done")


# # 卷积特征图的形状

# In[6]:

print ("Shape of 'train_features' is %s" % (train_features.shape,))
print ("Shape of 'test_features' is %s" % (test_features.shape,))


# # 向量化 

# In[7]:

# Vectorize
train_vectorized = np.ndarray((ntrain, 4*4*512))
test_vectorized  = np.ndarray((ntest, 4*4*512))
for i in range(ntrain):
    curr_feat = train_features[i, :, :, :]
    curr_feat_vec = np.reshape(curr_feat, (1, -1))
    train_vectorized[i, :] = curr_feat_vec
for i in range(ntest):
    curr_feat = test_features[i, :, :, :]
    curr_feat_vec = np.reshape(curr_feat, (1, -1))
    test_vectorized[i, :] = curr_feat_vec
    
print ("Shape of 'train_vectorized' is %s" % (train_features.shape,))
print ("Shape of 'test_vectorized' is %s" % (test_features.shape,))


# # 定义finetuning的结构

# In[8]:

# Parameters
learning_rate   = 0.0001
training_epochs = 100
batch_size      = 100
display_step    = 10
# tf Graph input
x = tf.placeholder(tf.float32, [None, 4*4*512])
y = tf.placeholder(tf.float32, [None, nclass])
keepratio = tf.placeholder(tf.float32)
# Network
with tf.device("/cpu:0"):
    n_input  = dim
    n_output = nclass
    weights  = {
        'wd1': tf.Variable(tf.random_normal([4*4*512, 1024], stddev=0.1)),
        'wd2': tf.Variable(tf.random_normal([1024, n_output], stddev=0.1))
    }
    biases   = {
        'bd1': tf.Variable(tf.random_normal([1024], stddev=0.1)),
        'bd2': tf.Variable(tf.random_normal([n_output], stddev=0.1))
    }
    def conv_basic(_input, _w, _b, _keepratio):
        # Input
        _input_r = _input
        # Vectorize
        _dense1 = tf.reshape(_input_r, [-1, _w['wd1'].get_shape().as_list()[0]])
        # Fc1
        _fc1 = tf.nn.relu(tf.add(tf.matmul(_dense1, _w['wd1']), _b['bd1']))
        _fc_dr1 = tf.nn.dropout(_fc1, _keepratio)
        # Fc2
        _out = tf.add(tf.matmul(_fc_dr1, _w['wd2']), _b['bd2'])
        # Return everything
        out = {'input_r': _input_r, 'dense1': _dense1,
            'fc1': _fc1, 'fc_dr1': _fc_dr1, 'out': _out }
        return out
    # Functions! 
    _pred = conv_basic(x, weights, biases, keepratio)['out']
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=_pred, labels=y))
    optm = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    _corr = tf.equal(tf.argmax(_pred,1), tf.argmax(y,1)) 
    accr = tf.reduce_mean(tf.cast(_corr, tf.float32)) 
    init = tf.initialize_all_variables()

print ("Network Ready to Go!")


# # 优化

# In[9]:

# Launch the graph
sess = tf.Session()
sess.run(init)

# Training cycle
for epoch in range(training_epochs):
    avg_cost = 0.
    num_batch = int(ntrain/batch_size)+1
    # Loop over all batches
    for i in range(num_batch): 
        randidx  = np.random.randint(ntrain, size=batch_size)
        batch_xs = train_vectorized[randidx, :]
        batch_ys = trainlabel[randidx, :]                
        # Fit training using batch data
        sess.run(optm, feed_dict={x: batch_xs, y: batch_ys, keepratio:0.7})
        # Compute average loss
        avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keepratio:1.})/num_batch

    # Display logs per epoch step
    if epoch % display_step == 0:
        print ("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
        train_acc = sess.run(accr, feed_dict={x: batch_xs, y: batch_ys, keepratio:1.})
        print (" Training accuracy: %.3f" % (train_acc))
        test_acc = sess.run(accr, feed_dict={x: test_vectorized, y: testlabel, keepratio:1.})
        print (" Test accuracy: %.3f" % (test_acc))

print ("Optimization Finished!")

