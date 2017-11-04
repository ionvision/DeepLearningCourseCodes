# Deep Learning Course Codes
Notes, Codes, and Tutorials for the Deep Learning Course at ChinaHadoop

> 注意每一份代码分别有Jupyter Notebook, Python, 以及HTML三种形式，大家可以按照自己的需求阅读，学习或运行。
> 运行时需要注意anaconda的版本问题，anaconda2-5.0.0与anaconda3-5.0.0分别对应python2.7与python3.6环境。

> 重要参考资料：
>    1. [Stanford CS229 Machine Learning, Fall 2017](http://cs229.stanford.edu/)
>    1. [Deep Learning Book读书笔记](https://github.com/exacity/simplified-deeplearning.git)
>    1. [Hands-on Machine Learning with Scikit-Learn and TensorFlow](https://github.com/ageron/handson-ml) [[book](https://www.amazon.com/_/dp/1491962291?tag=oreilly20-20)]

> 学习资料：
>    1. [Effective TensorFlow](https://github.com/vahidk/EffectiveTensorflow) - TensorFlow tutorials and best practices.
>    1. [Finch](https://github.com/zhedongzheng/finch) - Many Machine Intelligence models implemented (mainly tensorflow, sometimes pytorch / mxnet)
>    1. [Pytorch Tutorials](https://github.com/yunjey/pytorch-tutorial) - PyTorch Tutorial for Deep Learning Researchers.
>    1. [MXNet the straight dope](https://github.com/zackchase/mxnet-the-straight-dope) - An interactive book on deep learning. Much easy, so MXNet. Wow.

### 第一讲：深度学习课程总览与神经网络入门
> 代码示例：[TensorFlow基础与线性回归模型](https://github.com/jastarex/DeepLearningCourseCodes/tree/master/01_TF_basics_and_linear_regression)(TensorFlow, PyTorch)
> - [MNIST数据集演示](https://github.com/jastarex/DeepLearningCourseCodes/blob/master/01_TF_basics_and_linear_regression/mnist_data_introduction_tf.ipynb)
> - [TensorFlow基础](https://github.com/jastarex/DeepLearningCourseCodes/blob/master/01_TF_basics_and_linear_regression/tensorflow_basic.ipynb)
> - [线性回归模型-TensorFlow](https://github.com/jastarex/DeepLearningCourseCodes/blob/master/01_TF_basics_and_linear_regression/linear_regression_tf.ipynb)
> - [线性回归模型-PyTorch](https://github.com/jastarex/DeepLearningCourseCodes/blob/master/01_TF_basics_and_linear_regression/linear_regression_pt.ipynb)
> - [线性回归模型-MXNet](https://github.com/jastarex/DeepLearningCourseCodes/blob/master/01_TF_basics_and_linear_regression/linear_regression_mx.ipynb) (contributed by [LinkHS](https://github.com/LinkHS))

### 第二讲：传统神经网络
> 代码示例：[K近邻算法，线性分类，以及多层神经网络](https://github.com/jastarex/DeepLearningCourseCodes/tree/master/02_Logistic_regression_and_multilayer_perceptron)(TensorFlow, PyTorch)
> - [K近邻算法在图像分类上的应用-TensorFlow](https://github.com/jastarex/DeepLearningCourseCodes/blob/master/02_Logistic_regression_and_multilayer_perceptron/nearest_neighbor_tf.ipynb)
> - [K近邻算法在图像分类上的应用-PyTorch](https://github.com/jastarex/DeepLearningCourseCodes/blob/master/02_Logistic_regression_and_multilayer_perceptron/nearest_neighbor_pt.ipynb) (contributed by Johnny Chen)
> - [多层神经网络示例-TensorFlow](https://github.com/jastarex/DeepLearningCourseCodes/blob/master/02_Logistic_regression_and_multilayer_perceptron/neural_network_tf.ipynb)
> - [多层神经网络示例-PyTorch](https://github.com/jastarex/DeepLearningCourseCodes/blob/master/02_Logistic_regression_and_multilayer_perceptron/neural_network_pt.ipynb)

### 第三讲：卷积神经网络基础
> 代码示例：[卷积神经网络的基础实现](https://github.com/jastarex/DeepLearningCourseCodes/tree/master/03_CNN_basics)(TensorFlow)
> - [卷积神经网络基础示例-原生实现](https://github.com/jastarex/DeepLearningCourseCodes/blob/master/03_CNN_basics/cnn_tf_raw.ipynb)
> - [卷积神经网络基础示例-主流实现](https://github.com/jastarex/DeepLearningCourseCodes/blob/master/03_CNN_basics/cnn_tf.ipynb)

### 第四讲：卷积神经网络进阶
> 代码示例：[卷积神经网络的进阶实现](https://github.com/jastarex/DeepLearningCourseCodes/tree/master/04_CNN_advances)(TensorFlow)
> - [卷积神经网络进阶示例与可视化](https://github.com/jastarex/DeepLearningCourseCodes/blob/master/04_CNN_advances/cnn_mnist_simple.ipynb)
> - [卷积神经网络进阶示例-TF-Slim实现](https://github.com/jastarex/DeepLearningCourseCodes/blob/master/04_CNN_advances/cnn_mnist_modern.ipynb)

> - [准备自定义数据集训练卷积神经网络](https://github.com/jastarex/DeepLearningCourseCodes/blob/master/04_CNN_advances/basic_gendataset.ipynb)
> - [使用自定义数据集训练卷积神经网络](https://github.com/jastarex/DeepLearningCourseCodes/blob/master/04_CNN_advances/cnn_custom_simple.ipynb)

> - [使用训练好的VGG网络模型](https://github.com/jastarex/DeepLearningCourseCodes/blob/master/04_CNN_advances/use_vgg.ipynb)
> - [使用自定义数据集训练VGG网络模型](https://github.com/jastarex/DeepLearningCourseCodes/blob/master/04_CNN_advances/use_vgg_finetune.ipynb)

> - [基于TensorBoard的CNN可视化](https://github.com/jastarex/DeepLearningCourseCodes/blob/master/04_CNN_advances/vis_cnn_mnist.ipynb)

> - [更多基于TF-Slim的预训练模型](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models)

### 第五讲：深度神经网络：目标分类与识别
> 代码示例：[深度神经网络-图像识别与分类](https://github.com/jastarex/DeepLearningCourseCodes/tree/master/05_Image_recognition_and_classification)(TensorFlow, PyTorch)

- 安装[TensorLayer](https://github.com/zsdonghao/tensorlayer) (中文文档参见[此处](https://tensorlayercn.readthedocs.io/zh/latest/)，此后复杂实现均推荐使用TensorLayer高级API库，同时可以结合[TF-Slim](http://tensorlayercn.readthedocs.io/zh/latest/modules/layers.html#tf-slim)与[Keras](http://tensorlayercn.readthedocs.io/zh/latest/modules/layers.html#keras))
```
pip install git+https://github.com/zsdonghao/tensorlayer.git
```
- 安装[OpenCV](http://opencv.org/) python接口
```
conda install -c menpo opencv3 
或
pip install opencv-python
```
- 所需数据集下载：`data.zip`: [[微云](https://share.weiyun.com/7d008fcb693823503155acfc2be6ad2b)][[百度云](https://pan.baidu.com/s/1qYDhN5M)] (覆盖`./05_Image_recognition_and_classification/data`文件夹)  
- 所需模型下载： `vgg19.npz`  [[微云](https://share.weiyun.com/9fe52101fad44dadd4385d1f3d1e5804)][[百度云](https://pan.baidu.com/s/1qXIXr32)] (放置于`./05_Image_recognition_and_classification`文件夹下)  
- 所需模型下载：`inception_v3.ckpt` [[微云](https://share.weiyun.com/efdcea495ff2abd9cf271005a1d6f6b9)][[百度云](https://pan.baidu.com/s/1hrMB0Ug)] (放置于`./05_Image_recognition_and_classification`文件夹下) 

> - [图像识别与分类示例教程](https://github.com/jastarex/DeepLearningCourseCodes/blob/master/05_Image_recognition_and_classification/cnn.ipynb)

> - [VGG19图像分类模型-TensorLayer实现](https://github.com/jastarex/DeepLearningCourseCodes/blob/master/05_Image_recognition_and_classification/vgg19.ipynb)

> - [InceptionV3图像分类模型-TensorLayer结合TF-SLim实现](https://github.com/jastarex/DeepLearningCourseCodes/blob/master/05_Image_recognition_and_classification/inceptionV3_tfslim.ipynb)

> - [Wide-ResNet网络模型-TensorLayer实现](https://github.com/jastarex/DeepLearningCourseCodes/blob/master/05_Image_recognition_and_classification/wide_resnet_cifar.ipynb)

> - [Class Activation Mapping (CAM)示例](https://github.com/jastarex/DeepLearningCourseCodes/blob/master/05_Image_recognition_and_classification/pytorch_CAM.py) (完整实现可参考[此处](https://github.com/metalbubble/CAM))

### 第六讲：深度神经网络：目标检测与定位
> 代码示例：[目标检测模型示例](https://github.com/jastarex/DeepLearningCourseCodes/tree/master/01_TF_basics_and_linear_regression) (TensorFlow, PyTorch)

- [TensorFlow Object Detection API使用示例](https://github.com/jastarex/DeepLearningCourseCodes/tree/master/06_Object_detection/Object_Detection_Tensorflow_API_demo)

- 所需模型下载：`ssd_mobilenet_v1_coco_11_06_2017`: [[微云](https://share.weiyun.com/800e541b4403b07fb460fc017c77dc20)] (解压并置于`06_Object_detection/Object_Detection_Tensorflow_API_demo/object_detection/`文件夹下)

- [`SSD: Single Shot Multibox Detector`] ([TensorFlow实现](https://github.com/balancap/SSD-Tensorflow), [PyTorch实现](https://github.com/amdegroot/ssd.pytorch))

- [`YOLO`, `YOLOv2`] ([TensorFlow实现](https://github.com/ruiminshen/yolo-tf), [PyTorch实现](https://github.com/marvis/pytorch-yolo2))

### 第七讲：深度神经网络：目标追踪与目标分割
> 代码示例：[目标追踪与目标分割](https://github.com/jastarex/DeepLearningCourseCodes/tree/master/07_Object_tracking_and_object_segmentation)

- 目标追踪 - [`GOTURN`]([TensorFlow实现](https://github.com/tangyuhao/GOTURN-Tensorflow), [Plain Python实现](https://github.com/nrupatunga/PY-GOTURN), [Original C++实现](https://github.com/davheld/GOTURN))

- 目标分割 - [`FCN`]([TensorFlow实现](https://github.com/shekkizh/FCN.tensorflow), [PyTorch实现](https://github.com/wkentaro/pytorch-fcn))

- 目标分割 - [`Mask-RCNN`]([TensorFlow实现](https://github.com/CharlesShang/FastMaskRCNN), [PyTorch实现](https://github.com/felixgwu/mask_rcnn_pytorch))

### 第八讲：循环神经网络与序列模型
> 代码示例：[循环神经网络](https://github.com/jastarex/DeepLearningCourseCodes/tree/master/08_RNN_and_Seq2Seq)

- 循环神经网络, `RNN` - [[TensorFlow](https://github.com/jastarex/DeepLearningCourseCodes/blob/master/08_RNN_and_Seq2Seq/rnn_tf.ipynb), [Pytorch](https://github.com/jastarex/DeepLearningCourseCodes/blob/master/08_RNN_and_Seq2Seq/rnn_pytorch.py)]

- 双向循环神经网络, `Bidirectional-RNN` - [[TensorFlow](https://github.com/jastarex/DeepLearningCourseCodes/blob/master/08_RNN_and_Seq2Seq/bidirectional_rnn_tf.ipynb), [Pytorch](https://github.com/jastarex/DeepLearningCourseCodes/blob/master/08_RNN_and_Seq2Seq/bidirection_rnn_pytorch.py)]

- 动态循环神经网络, `Dynamic-RNN` - [[TensorFlow](https://github.com/jastarex/DeepLearningCourseCodes/blob/master/08_RNN_and_Seq2Seq/dynamic_rnn_tf.ipynb)]

- 自动编码器, `AutoEncoder` - [[TensorFlow](https://github.com/jastarex/DeepLearningCourseCodes/blob/master/08_RNN_and_Seq2Seq/autoencoder_tf.ipynb)]

- 变分自动编码器, `Variational AutoEncoder` - [[TensorFlow](https://github.com/jastarex/DeepLearningCourseCodes/blob/master/08_RNN_and_Seq2Seq/variational_autoencoder_tf.ipynb)]

- 图片标注, `Image Captioning` - [[TensorFlow](https://github.com/zsdonghao/Image-Captioning), [PyTorch](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning)]

- 视频标注, `Video Captioning` - [[TensorFlow](https://github.com/chenxinpeng/S2VT)]

### 第九讲：无监督式学习与生成对抗网络
> 代码示例：[生成对抗网络](https://github.com/jastarex/DeepLearningCourseCodes/tree/master/09_Unsupervised_learning_and_GANs)

- 生成对抗网络, `GAN` - [[TensorFlow](https://github.com/jastarex/DeepLearningCourseCodes/blob/master/09_Unsupervised_learning_and_GANs/gan.ipynb), [PyTorch](https://github.com/jastarex/DeepLearningCourseCodes/blob/master/09_Unsupervised_learning_and_GANs/gan_pytorch.py)] 

- 深度卷积生成对抗网络, `DCGAN` - [[TensorFlow](https://github.com/jastarex/DeepLearningCourseCodes/blob/master/09_Unsupervised_learning_and_GANs/dcgan.ipynb)]

- `Pix2Pix`, `CycleGAN` - [[PyTorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)]

- `StackGAN` - [[TensorFlow, PyTorch](https://github.com/hanzhanggit/StackGAN)]

- `GANs Model Collections` - [[1](https://github.com/wiseodd/generative-models), [2](https://github.com/hwalsuklee/tensorflow-generative-model-collections), [3](https://github.com/znxlwm/pytorch-generative-model-collections)]

### 第十讲：强化学习

- `Basic Reinforcement Learning` - [[basic_reinforcement_learning](https://github.com/vmayoral/basic_reinforcement_learning)]

- `Applied Reinfocement Learning` - [[applied-reinforcement-learning](https://github.com/mimoralea/applied-reinforcement-learning)]

- `Oreilly RL Tutorial` - [[oreilly-rl-tutorial](https://github.com/awjuliani/oreilly-rl-tutorial)]
