import tensorflow as tf
import warnings
import logging
import time


# tf dec fun can also take tensor as param
@tf.function
def add(a, b):
    return a + b
@tf.function
def sub(a,b):
    return a - b

@tf.function
def mul(a, b):
    return a * b

@tf.function
def div(a, b):
    return a / b


print(add(tf.constant(5), tf.constant(2)))


print(sub(tf.constant(5), tf.constant(2)))


print(mul(tf.constant(5), tf.constant(2)))


print(div(tf.constant(5), tf.constant(2)))


# tf decorated functino invoking another natural tf function
@tf.function
def matmul(a, b):
    return tf.matmul(a, b)

@tf.function
def linear(m, x, c):
    return add(matmul(m, c), c)


m = tf.constant([[4.0], [5.0], [6.0]], tf.float32)

m


x = tf.Variable([[100.0], [100.0], [100.0]], tf.float32)

x


c = tf.constant([[1.0]], tf.float32)

c


linear(m, x, c)


# tf.fun, coverts regular py dyanmic constructs like if, for to tf equivalent
@tf.function
def pos_neg_check(x):
    reduce_sum = tf.reduce_sum(x)  # computes sum of elements of input x
    
    if(reduce_sum > 0):
        return tf.constant(1)
    elif(reduce_sum == 0):
        return tf.constant(0)
    else:
        return tf.constant(-1)


pos_neg_check(tf.constant([100, 100]))


num = tf.Variable(7)


@tf.function
def add_times(x):
    for i in tf.range(x):  # tf.range allows unrulling of for loop in stat computation graph.
        num.assign_add(x)
        
add_times(5)


num


@tf.function
def square(a):
    print("input a: ", a)
    return a * a


x = tf.Variable([[2, 2], [2, 2]], tf.float32)

square(x)


y = tf.Variable([[2, 2], [2, 2]], tf.int32)

square(y)


z = tf.Variable([[3, 3], [3, 3]], tf.float32)

square(z)


# concrete fun for int32 dtype
concrete_int_square_fn = square.get_concrete_function(tf.TensorSpec(shape = None, dtype = tf.int32))

concrete_int_square_fn  # works for only int32 params


concrete_float_square_fn = square.get_concrete_function(tf.TensorSpec(shape = None, dtype = tf.float32))

concrete_float_square_fn


concrete_int_square_fn(tf.constant([[2, 2], [2, 2]], dtype = tf.int32))


@tf.function
def f(x):
    print("python exec: ", x)  # side effect statement exec only once, first time fn called i.e when graph is traced if same data
    tf.print("Graph exec: ", x)


f(1)


f(1)


f("Hello tf.functino")


def fn_with_variable_init_eager():
    a = tf.constant([[10, 10], [11., 1.]])
    x= tf.constant([[1., 0.], [0., 1.]])
    b = tf.Variable(12.)
    
    y = tf.matmul(a, x) + b
    
    tf.print("tf_print: ", y)
    return y


fn_with_variable_init_eager()


# auth graph(lazy exec)
@tf.function
def fn_with_variable_init_autograph():
    a = tf.constant([[10, 10], [11., 1.]])
    x= tf.constant([[1., 0.], [0., 1.]])
    b = tf.Variable(12.)
    
    y = tf.matmul(a, x) + b
    
    tf.print("tf_print: ", y)
    return y


fn_with_variable_init_autograph()  # gives error since, tried to create a var again after previously created(not allowed in graph mode)


class F():
    def __init__(self):
        self._b = None
        
    @tf.function
    def __call__(self):
        a = tf.constant([[10, 10], [11., 1.]])
        x= tf.constant([[1., 0.], [0., 1.]])
        
        if self._b is None:  # create a variable only first time fun is called
            self._b = tf.Variable(12.)
            
        y = tf.matmul(a, x) + self._b
        print(y)
        
        tf.print("tf_print: ", y)
        return y
    
fn_with_variable_init_autograph = F()
fn_with_variable_init_autograph()


# allows us to see equi graph code of normal fun
def f(x):
    if x > 0:
        x *= x
    return x

print(tf.autograph.to_code(f))


































































