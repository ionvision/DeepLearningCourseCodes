
# coding: utf-8

# # TensorFlow基础
# In this tutorial, we are going to learn some basics in TensorFlow.

# ## Session
# Session is a class for running TensorFlow operations. A Session object encapsulates the environment in which Operation objects are executed, and Tensor objects are evaluated. In this tutorial, we will use a session to print out the value of tensor. Session can be used as follows:

# In[1]:

import tensorflow as tf

a = tf.constant(100)

with tf.Session() as sess:
    print sess.run(a)
    #syntactic sugar
    print a.eval()
    
# or

sess = tf.Session()
print sess.run(a)
# print a.eval()     # this will print out an error


# ## Interactive session
# Interactive session is a TensorFlow session for use in interactive contexts, such as a shell. The only difference with a regular Session is that an Interactive session installs itself as the default session on construction. The methods [Tensor.eval()](https://www.tensorflow.org/versions/r0.11/api_docs/python/framework.html#Tensor) and [Operation.run()](https://www.tensorflow.org/versions/r0.11/api_docs/python/framework.html#Operation) will use that session to run ops.This is convenient in interactive shells and IPython notebooks, as it avoids having to pass an explicit Session object to run ops.

# In[2]:

sess = tf.InteractiveSession()

print a.eval()    # simple usage  


# ## Constants
# We can use the `help` function to get an annotation about any function. Just type `help(tf.consant)` on the below cell and run it.
# It will print out `constant(value, dtype=None, shape=None, name='Const')` at the top. Value of tensor constant can be scalar, matrix or tensor (more than 2-dimensional matrix). Also, you can get a shape of tensor by running [tensor.get_shape()](https://www.tensorflow.org/versions/r0.11/api_docs/python/framework.html#Tensor)`.as_list()`. 
# 
# * tensor.get_shape()
# * tensor.get_shape().as_list()

# In[3]:

a = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32, name='a')
print a.eval()
print "shape: ", a.get_shape(), ",type: ", type(a.get_shape())
print "shape: ", a.get_shape().as_list(), ",type: ", type(a.get_shape().as_list())   # this is more useful


# ## Basic functions
# There are some basic functions we need to know. Those functions will be used in next tutorial **3. feed_forward_neural_network**.
# * tf.argmax
# * tf.reduce_sum
# * tf.equal
# * tf.random_normal

# #### tf.argmax 
# `tf.argmax(input, dimension, name=None)` returns the index with the largest value across dimensions of a tensor.
# 

# In[4]:

a = tf.constant([[1, 6, 5], [2, 3, 4]])
print a.eval()
print "argmax over axis 0"
print tf.argmax(a, 0).eval()
print "argmax over axis 1"
print tf.argmax(a, 1).eval()


# #### tf.reduce_sum
# `tf.reduce_sum(input_tensor, reduction_indices=None, keep_dims=False, name=None)` computes the sum of elements across dimensions of a tensor. Unless `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in reduction_indices. If `keep_dims` is true, the reduced dimensions are retained with length 1. If `reduction_indices` has no entries, all dimensions are reduced, and a tensor with a single element is returned

# In[5]:

a = tf.constant([[1, 1, 1], [2, 2, 2]])
print a.eval()
print "reduce_sum over entire matrix"
print tf.reduce_sum(a).eval()
print "reduce_sum over axis 0"
print tf.reduce_sum(a, 0).eval()
print "reduce_sum over axis 0 + keep dimensions"
print tf.reduce_sum(a, 0, keep_dims=True).eval()
print "reduce_sum over axis 1"
print tf.reduce_sum(a, 1).eval()
print "reduce_sum over axis 1 + keep dimensions"
print tf.reduce_sum(a, 1, keep_dims=True).eval()


# #### tf.equal
# `tf.equal(x, y, name=None)` returns the truth value of `(x == y)` element-wise. Note that `tf.equal` supports broadcasting. For more about broadcasting, please see [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).

# In[6]:

a = tf.constant([[1, 0, 0], [0, 1, 1]])
print a.eval()
print "Equal to 1?"
print tf.equal(a, 1).eval()
print "Not equal to 1?"
print tf.not_equal(a, 1).eval()


# #### tf.random_normal
# `tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)` outputs random values from a normal distribution.
# 

# In[7]:

normal = tf.random_normal([3], stddev=0.1)
print normal.eval()


# ## Variables
# When we train a model, we use variables to hold and update parameters. Variables are in-memory buffers containing tensors. They must be explicitly initialized and can be saved to disk during and after training. we can later restore saved values to exercise or analyze the model.
# 
# * tf.Variable
# * tf.Tensor.name
# * tf.all_variables
# 

# #### tf.Variable
# `tf.Variable(initial_value=None, trainable=True, name=None, variable_def=None, dtype=None)` creates a new variable with value `initial_value`.
# The new variable is added to the graph collections listed in collections, which defaults to `[GraphKeys.VARIABLES]`. If `trainable` is true, the variable is also added to the graph collection `GraphKeys.TRAINABLE_VARIABLES`. 

# In[8]:

# variable will be initialized with normal distribution
var = tf.Variable(tf.random_normal([3], stddev=0.1), name='var')
print var.name
tf.initialize_all_variables().run()
print var.eval()


# #### tf.Tensor.name
# We can call `tf.Variable` and give the same name `my_var` more than once as seen below. Note that `var3.name` prints out `my_var_1:0` instead of `my_var:0`. This is because TensorFlow doesn't allow user to create variables with the same name. In this case, TensorFlow adds `'_1'` to the original name instead of printing out an error message. Note that you should be careful not to call `tf.Variable` giving same name more than once, because it will cause a fatal problem when you save and restore the variables.

# In[9]:

var2 = tf.Variable(tf.random_normal([2, 3], stddev=0.1), name='my_var')
var3 = tf.Variable(tf.random_normal([2, 3], stddev=0.1), name='my_var')
print var2.name
print var3.name


# #### tf.all_variables
# Using `tf.all_variables()`, we can get the names of all existing variables as follows:

# In[10]:

for var in tf.all_variables():
    print var.name


# ## Sharing variables
# TensorFlow provides several classes and operations that you can use to create variables contingent on certain conditions.
# * tf.get_variable
# * tf.variable_scope
# * reuse_variables

# #### tf.get_variable
# `tf.get_variable(name, shape=None, dtype=None, initializer=None, trainable=True)` is used to get or create a variable instead of a direct call to `tf.Variable`. It uses an initializer instead of passing the value directly, as in `tf.Variable`. An initializer is a function that takes the shape and provides a tensor with that shape. Here are some initializers available in TensorFlow:
# 
# * `tf.constant_initializer(value)` initializes everything to the provided value,
# * `tf.random_uniform_initializer(a, b)` initializes uniformly from [a, b],
# * `tf.random_normal_initializer(mean, stddev)` initializes from the normal distribution with the given mean and standard deviation.

# In[11]:

my_initializer = tf.random_normal_initializer(mean=0, stddev=0.1)
v = tf.get_variable('v', shape=[2, 3], initializer=my_initializer)
tf.initialize_all_variables().run()
print v.eval()


# #### tf.variable_scope
# `tf.variable_scope(scope_name)` manages namespaces for names passed to `tf.get_variable`.

# In[12]:

with tf.variable_scope('layer1'):
    w = tf.get_variable('v', shape=[2, 3], initializer=my_initializer)
    print w.name
    
with tf.variable_scope('layer2'):
    w = tf.get_variable('v', shape=[2, 3], initializer=my_initializer)
    print w.name


# #### reuse_variables
# Note that you should run the cell above only once. If you run the code above more than once, an error message will be printed out: `"ValueError: Variable layer1/v already exists, disallowed."`. This is because we used `tf.get_variable` above, and this function doesn't allow creating variables with the existing names. We can solve this problem by using `scope.reuse_variables()` to get preivously created variables instead of creating new ones. 

# In[13]:

with tf.variable_scope('layer1', reuse=True):
    w = tf.get_variable('v')   # Unlike above, we don't need to specify shape and initializer 
    print w.name

# or

with tf.variable_scope('layer1') as scope:
    scope.reuse_variables()
    w = tf.get_variable('v')
    print w.name


# ## Place holder
# TensorFlow provides a placeholder operation that must be fed with data on execution. If you want to get more details about placeholder, please see [here](https://www.tensorflow.org/versions/r0.11/api_docs/python/io_ops.html#placeholder).

# In[14]:

x = tf.placeholder(tf.int16)
y = tf.placeholder(tf.int16)

add = tf.add(x, y)
mul = tf.mul(x, y)

# Launch default graph.
print "2 + 3 = %d" % sess.run(add, feed_dict={x: 2, y: 3})
print "3 x 4 = %d" % sess.run(mul, feed_dict={x: 3, y: 4})

