import tensorflow as tf
import numpy as np
tf.debugging.set_log_device_placement(True)   # to print log msg about where logs are located and on which devices they are exec


tf.executing_eagerly()


# note: value of a constant is immutable unlike variables which is not
x0 = tf.constant(3)

x0


x0.shape  # gives nothing since x0 is scalar


x0.dtype


x0.numpy


result0 = x0 + 5

result0  # numpy = 8 is it's value


# vector
x1 = tf.constant([1.1, 2.2, 3.3, 4.4])
x1


result1 = x1 + 5

result1


result1 = x1 + tf.constant(5.0)

result1


result1 = tf.add(x1, tf.constant(5.0))

result1


x2 = tf.constant([[1, 2, 3, 4],
                 [5, 6, 7, 8]])

x2


# change dtype of a tensor
x2 = tf.cast(x2, tf.float32)

x2


# to perform *, ensure shapes based on matrix rule match
result3 = tf.multiply(x1, x2)

result3


# conv tensor it np equivalent array
arr_x1 = x1.numpy()

arr_x1


arr_x4 = np.array([[10, 20], [30, 40], [50, 60]])

arr_x4


# conv np to tensor
x4 = tf.convert_to_tensor(arr_x4)

x4


# np operation on tensor, but not recommended
np.square(x2)


# check if var is a tensor
tf.is_tensor(arr_x4)


# tensor of zero val
t0 = tf.zeros([3, 5], tf.int32)

t0


t1 = tf.ones([3, 5], tf.int32)

t1


# reshaping tensor
t0_reshaped = tf.reshape(t0, (5, 3))

t0_reshaped


#n.b:  tensor stored in a variable, can be changed by running operations on that var

v1 = tf.Variable([[1.5, 2, 5], [2, 6, 8]])

v1


v2 = tf.Variable([[1, 2, 3], [4, 5, 6]], dtype = tf.float32)

v2


tf.add(v1, v2)


# view tensor rep of variable
tf.convert_to_tensor(v1)


v1.numpy()


v1


v1.assign([[10, 20, 30], [40, 50, 60]])

v1


v1[0, 0].assign(100)

v1


v1.assign_add([[1, 1, 1], [1, 1, 1]])

v1


v1.assign_sub([[3, 3, 3], [3, 3, 3]])

v1


var_a = tf.Variable([2.0, 3.0])

var_a


# creating a variable using a copy of another var
var_b = tf.Variable(var_a)

var_b


var_b.assign([200, 500])

var_b


var_a












































































































