import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# actual correct value of bias and W(weight)
W_true = 2
b_true = 0.5


x = np.linspace(0, 3, 130)

y = W_true * x + b_true + np.random.randn(*x.shape) * 0.5


# data viz our gen data
plt.figure(figsize = (8, 8))

plt.scatter(x, y)
plt.xlabel("x")
plt.ylabel("y")

plt.title("Training Data")
plt.show()


# class to init w and b to be modified during training
class LinearModel:
    
    def __init__(self):
        self.weight = tf.Variable(np.random.randn(), name = "w")
        self.bias = tf.Variable(np.random.randn(), name ="b")
        
    def __call__(self, x):  # called during forward pass of simple linreg model
        return self.weight * x + self.bias


def loss(y, y_pred):
    return tf.reduce_mean(tf.square(y - y_pred))


# training process of lin model
def train(linear_model, x, y, lr = 0.01):
    with tf.GradientTape() as tape:
        
        y_pred = linear_model(x)  # passing x calls the __call__ fun of linear model class after init
        
        current_loss = loss(y, y_pred)
        
    # calc gradient of current loss wrt weight and bias(trainable params)
    d_weight, d_bias = tape.gradient(current_loss, [linear_model.weight, linear_model.bias])
    
    # updates weight value and bias value by subing lr* d_weight and d_bias from previous weight value
    linear_model.weight.assign_sub(lr * d_weight)
    linear_model.bias.assign_sub(lr * d_bias)


linear_model = LinearModel()

weights, biases = [], []

epochs = 10

lr = 0.15


for epoch_count in range(epochs):
    weights.append(linear_model.weight.numpy())
    biases.append(linear_model.bias.numpy())
    
    real_loss = loss(y, linear_model(x))
    
    train(linear_model, x, y, lr = lr)
    
    print(f"Epoch count {epoch_count}: Loss Value: {real_loss.numpy()}")


# data viz on how w and b of trained model, match up with actau; w and b

plt.figure(figsize = (8, 8))

plt.plot(range(epochs), weights, "r", range(epochs), biases, "b")
plt.plot([W_true] * epochs, "r--", [b_true] * epochs, "b--")

plt.legend(["w", "b", "true W", "true b"])
plt.show();

# dotted lines is true value of w and b


# final w and b gotten via the model
linear_model.weight, linear_model.bias


# mse
rmse = loss(y, linear_model(x))

rmse.numpy()


# viz original data and linear model as a fitted line
plt.figure(figsize = (8, 8))

plt.plot(x, y, "ro", label = "Original data")
plt.plot(x, linear_model(x), label = "fitted line")
plt.title("Linear regression")
plt.legend()
plt.show()


from tensorflow import keras
from tensorflow.keras import layers


x.shape, y.shape


x = pd.DataFrame(x, columns = ["x"])
y = pd.DataFrame(y, columns = ["y"])

y.head()


x.shape


model = keras.Sequential([
    layers.Dense(1, input_shape = (1,), activation="linear")
])

optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
model.compile(loss = "mse", metrics = ["mse"], optimizer = optimizer)


model.fit(x, y, epochs = 100)


y_pred = model.predict(x)


# data viz
plt.figure(figsize = (10, 8))

plt.scatter(x, y, c = "blue", label = "Original data")
plt.plot(x, y_pred, color = "r", label="fitted line")

plt.title("Linear Regression")
plt.legend()
plt.show()







































