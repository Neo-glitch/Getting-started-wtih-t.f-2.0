import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow.keras.utils import to_categorical


wine_data = datasets.load_wine()


# description of wine dataset
print(wine_data["DESCR"])


# df of wine data
data = pd.DataFrame(data = wine_data.data, columns = wine_data.feature_names)

data["target"] = wine_data.target

data.sample(5)


data.describe().T


data.isna().sum()  # no na values


data.target.value_counts()


# find alcohol distro across wine in data
sns.distplot(data["alcohol"], kde = 1)


# alcohol content variation based on diff cat of wine
plt.figure(figsize = (10, 8))

sns.boxplot("target", "alcohol", data = data)

plt.xlabel("target", fontsize = 20)
plt.ylabel("alcohol", fontsize = 20)
plt.show()


features = data.drop("target", axis = 1)

target = data[["target"]]


features.columns


# since multi class best to one hot encode target
target = to_categorical(target, 3)

target


standradScaler = StandardScaler()

processed_features = pd.DataFrame(standradScaler.fit_transform(features),
                                 columns = features.columns,
                                 index = features.index)

processed_features.describe().T


x_train, x_test, y_train, y_test = train_test_split(processed_features, target,
                                                   test_size = 0.2, random_state = 1)


class WineClassificationModel(Model):   # class inherits from tf.keras model class
    
    def __init__(self, input_shape):
        super(WineClassificationModel, self).__init__()  # calls superclass(Model) init func
        
        self.d1 = layers.Dense(128, activation = "relu", input_shape = [input_shape])
        self.d2 = layers.Dense(64, activation = "relu")
        self.d3 = layers.Dense(3, activation = "softmax")
        
    def call(self, x):  # called in forward pass of this model
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        
        return x


model = WineClassificationModel(x_train.shape[1])

model.compile(optimizer = keras.optimizers.SGD(lr=0.001),
             loss = keras.losses.CategoricalCrossentropy(),  # used for multi class clf where target is 1-hot encoded
             metrics = ["accuracy"])


num_epochs = 500


training_history = model.fit(x_train.values,
                            y_train,
                            validation_split = 0.2,
                            epochs = num_epochs, batch_size = 40)


training_history.history.keys()


# data viz models performance
train_acc = training_history.history["accuracy"]
train_loss = training_history.history["loss"]

val_accuracy = training_history.history["val_accuracy"]
val_loss = training_history.history["val_loss"]

epochs_range = range(num_epochs)

plt.figure(figsize = (12, 8))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_acc, label = "Training_accuracy")
plt.plot(epochs_range, train_loss, label = "Training_loss")
plt.title("Training")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, val_accuracy, label = "Val_accuracy")
plt.plot(epochs_range, val_loss, label = "Val_loss")
plt.title("Validation")
plt.legend()


score = model.evaluate(x_test, y_test)

score_df = pd.Series(score, index = model.metrics_names)

score_df


y_pred= model.predict(x_test)

y_pred[:10]


# implementing thresholding on score i.e score < 0.5 = 0 and > 0.5 = 1
y_pred = np.where(y_pred <0.5, 0, y_pred)

y_pred = np.where(y_pred >=0.5, 1, y_pred)


y_pred[:10]


y_test[:10]


accuracy_score(y_test, y_pred)






















































