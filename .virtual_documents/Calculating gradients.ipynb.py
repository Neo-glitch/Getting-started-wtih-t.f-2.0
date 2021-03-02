import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


x = tf.Variable(4.0)

with tf.GradientTape() as tape:
    y = x**2   # gtape records all op that occurs in forward pass


y


dy_dx = tape.gradient(y, x)  #calc the gradients of y wrt input x(i.e if x changes how much does y change)

dy_dx


w = tf.Variable(tf.random.normal((4, 2)))

w


b = tf.Variable(tf.ones(2, dtype = tf.float32))

b


x = tf.Variable([[10., 20., 30., 40.]], dtype = tf.float32)

x


# persistent = true allows a gradient tape of same operations to be invoked multiple times
with tf.GradientTape(persistent = True) as tape:
    y = tf.matmul(x, w) + b
    
    loss = tf.reduce_mean(y**2) # use mean of values from y array as loss


[d1_dw, d1_db] = tape.gradient(loss, [w, b])    # calc gradient of loss wrt w and b


d1_dw


loss


# another test
layer = tf.keras.layers.Dense(2, activation = "relu")

x = tf.constant([[10., 20., 30.]])


with tf.GradientTape() as tape:
    y = layer(x)
    loss = tf.reduce_sum(y**2)
    
grad = tape.gradient(loss, layer.trainable_variables)


grad


x1 = tf.Variable(5.0)

x1


x2 = tf.Variable(5.0, trainable = False)  # non-trainable var

x2


x3 = tf.add(x1, x2)

x3


x4 = tf.constant(5.0)

x4


with tf.GradientTape() as tape:
    y = (x1**2) + (x2**2) + (x3**2) + (x4**2)
    
grad = tape.gradient(y, [x1, x2, x3, x4])  # calc gradient of y wrt x1, x2, x3, x4

grad #gives gradient value of x1(10.0) since others can't be tracked by default


# how to track constants, tensors e.t.c

x1 = tf.constant(5.0)

x2 = tf.Variable(3.0)


with tf.GradientTape(persistent = True) as tape:
    tape.watch(x1)  # allows constant to be tracked
    
    y = (x1**2) + (x2**2)
    
grad = tape.gradient(y, [x1, x2])

grad


# explicitly determing what var to track in gradient tape(tracks only what we watch using tape.watch)
with tf.GradientTape(watch_accessed_variables=False) as tape:
    tape.watch(x1) # tracks only x1
    
    y = (x1**2) + (x2**2)
    
grad = tape.gradient(y, [x1, x2])

grad


x = tf.constant(1.0)
x1 = tf.Variable(5.0)
x2 = tf.Variable(3.0)


with tf.GradientTape(persistent = True) as tape:
    tape.watch(x)
    
    # tracks only actual operation that was ran and not all ctrl flow block
    if(x > 0.0):  # this block operation will be tracked in this case since x > 0.0
        result = x1**2
    else:
        result = x2**2
        
dx1, dx2 = tape.gradient(result, [x1, x2])

dx1, dx2


x = tf.constant(-1.0)
x1 = tf.Variable(5.0)
x2 = tf.Variable(3.0)


with tf.GradientTape(persistent = True) as tape:
    tape.watch(x)
    
    # tracks only actual operation that was ran and not all ctrl flow block
    if(x > 0.0):
        result = x1**2
    else: # this block op will be tracked in this case
        result = x2**2
        
dx1, dx2 = tape.gradient(result, [x1, x2])

dx1, dx2 # x1 is None


























































































