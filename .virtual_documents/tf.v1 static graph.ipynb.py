import tensorflow as tf

# checks if we are in eager_exec mode(v2)
tf.compat.v1.executing_eagerly()


a = tf.constant(5, name = "a")
b = tf.constant(7, name = "b")

c = tf.add(a, b, name = "sum")


# to work as static grapgh, erroe since session has no graph to exec
# since at eager computation is exec right away
sess = tf.compat.v1.Session()

sess.run(c)


c


# disable eager exec to use static comp graph

tf.compat.v1.disable_eager_execution()

tf.compat.v1.executing_eagerly()


tf.compat.v1.reset_default_graph()  # clears graph


a = tf.constant(5, name = "a")
b = tf.constant(7, name = "b")

c = tf.add(a, b, name = "sum")


c  # not computed here since, in static graph mode


sess = tf.compat.v1.Session()

sess.run(c)


d = tf.multiply(a, b, name = "product")

sess.run(d)


# best practice when done with a session
sess.close()


m = tf.Variable([4.0, 5.0, 6.0], tf.float32, name = "m")

c = tf.Variable([1.0, 1.0, 1.0], tf.float32, name = "c")


m


# place holders used in tf to specify input in a nn(shape of input, but not actual value)
x = tf.compat.v1.placeholder(tf.float32, shape =[3], name = "x")

x


y = m * x + c
y


# init variables used in a graph
init = tf.compat.v1.global_variables_initializer()


# run session and write info to tf board
with tf.compat.v1.Session() as sess:
    sess.run(init)  # runs init to init var
    
    # feed dict is to supply values of X(placeholder)
    y_output = sess.run(y, feed_dict = {x: [100.0, 100.0, 100.0]})
    
    print("Final result: mx + c = ", y_output)
    
    writer = tf.compat.v1.summary.FileWriter("./logs/", sess.graph)  # writes the graph to tf baord
    writer.close()


# loading tf board
get_ipython().run_line_magic("reload_ext", " tensorboard")


# viewing tf board
get_ipython().run_line_magic("tensorboard", " --logdir=\"D:\PluralSight_ml_Courses\Getting Started with tf 2.0\logs\" --port 7070")


































































