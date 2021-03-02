import tensorflow as tf
tf.executing_eagerly()


x = [[10.]]

res = tf.matmul(x, x)

res


a = tf.constant([[10, 20],
                [30, 40]])
a


b = tf.add(a, 2)

b


print(a * b)


m = tf.Variable([4.0, 5.0, 6.0], tf.float32, name = "m")
c = tf.Variable([1.0, 1.0, 1.0], tf.float32, name = "c")


# since eager exec, place holder concept not exist
x = tf.Variable([100.0, 100.0, 100.0], tf.float32, name = "x")

x


y = m * x + c

y




































