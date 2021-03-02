import os, datetime

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# aim is to predict life expectancy of a country
data = pd.read_csv("./datasets/life_expectancy.csv")

data.sample(5)


data.shape


data.isna().sum()  # gets num of nan entris per col


# fill na value of a col with mean value of that col
countries = data["Country"].unique()


# cols that have null values
na_cols = ["Life expectancy ", "Adult Mortality", "Alcohol", "Hepatitis B",
          " BMI ", "Polio", "Total expenditure", "Diphtheria ", "GDP",
          " thinness  1-19 years", " thinness 5-9 years", "Population",
          "Income composition of resources"]

for col in na_cols:
    for country in countries:
        data.loc[data["Country"] == country, col] = data.loc[data["Country"] == country, col]\
                                                    .fillna(data[data["Country"] == country][col].mean())  # calc mean of col in focus and replace nan values with it
        


data.isna().sum()


# since still nan values(occurs if a col has all values to be nan)

data = data.dropna()  # drop all nan values

data.shape


# EDA
data["Status"].value_counts()


data["Country"].value_counts()


plt.figure(figsize = (10, 8))

data.boxplot('Life expectancy ')
plt.show()


plt.figure(figsize = (10, 8))

sns.boxplot("Status", 'Life expectancy ', data = data)  # shows that life expectancy is higher in developed countries.
plt.xlabel("Status", fontsize = 16)
plt.ylabel("Total expenditure", fontsize = 16)

plt.show()


data_corr = data[["Life expectancy ",
                 "Adult Mortality",
                 "Schooling",
                 "Total expenditure",
                 "Diphtheria ",
                 "GDP",
                 "Population"]].corr()

data_corr


fig, ax = plt.subplots(figsize = (10, 10))

sns.heatmap(data_corr, annot= True)

plt.show()


# split dataset into features and target
features = data.drop("Life expectancy ", axis = 1)

target = data[["Life expectancy "]]


features.columns


target.sample(5)


# drop country col(seems not useful)
features = features.drop("Country", axis = 1)

features.columns


# try to convert status(cat data) to num data
categorical_features = features["Status"].copy()

categorical_features.head()


categorical_features = pd.get_dummies(categorical_features)

categorical_features.head()


numeric_features = features.drop(["Status"], axis = 1)  # drop cat col(status) and assign to num_features var

numeric_features.head()


numeric_features.describe().T


# Standardize dataset
standardScaler = StandardScaler()

numeric_features = pd.DataFrame(standardScaler.fit_transform(numeric_features),
                               columns = numeric_features.columns,
                               index = numeric_features.index)

numeric_features.describe().T


# combine preprocessed cat data an num data
processed_features = pd.concat([numeric_features, categorical_features], axis = 1, sort = False)

processed_features.head()


processed_features.shape


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(processed_features,
                                                   target,
                                                   test_size = 0.2,
                                                   random_state = 1)


def build_single_layer_model():
    model = tf.keras.Sequential()
    
    model.add(tf.keras.layers.Dense(32,
                                    input_shape = (x_train.shape[1],), # input shape is num of features
                                   activation = "sigmoid"))
    model.add(tf.keras.layers.Dense(1))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)
    
    model.compile(loss = "mse", metrics = ["mae", "mse"],
                 optimizer = optimizer)
    
    return model


model = build_single_layer_model()

model.summary()


# viz model layers
tf.keras.utils.plot_model(model)


num_epochs = 100

training_history = model.fit(x_train, y_train,
                            epochs = num_epochs,
                            validation_split = 0.2,
                            verbose = True)


# model performance viz
plt.figure(figsize = (10, 10))
plt.subplot(1, 2, 1)

plt.plot(training_history.history["mae"])
plt.plot(training_history.history["val_mae"])

plt.title("Model MAE")
plt.ylabel("mae")
plt.xlabel("epoch")
plt.legend(["train", "val"])

plt.subplot(1, 2, 2)

plt.plot(training_history.history["loss"])
plt.plot(training_history.history["val_loss"])

plt.title("Model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "val"])


model.evaluate(x_test, y_test)


y_pred = model.predict(x_test)

r2_score(y_test, y_pred)


# df to hold actual life expectancy value and predicted life expectancy value
pred_results = pd.DataFrame({"y_test": y_test.values.flatten(),
                            "y_pred": y_pred.flatten()}, index = range(len(y_pred)))

pred_results.sample(10)


# build multi layered model
def build_multiple_layer_model():
    model = keras.Sequential([
        layers.Dense(2, input_shape = (x_train.shape[1],), activation = "relu"),
        layers.Dense(16, activation = "relu"),
        layers.Dense(4, activation = "relu"),
        layers.Dense(1)
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
    
    model.compile(loss = "mse", metrics = ["mae", "mse"], optimizer = optimizer)
    
    return model


model = build_multiple_layer_model()

tf.keras.utils.plot_model(model, show_shapes = True)


logdir = os.path.join("./logs/", datetime.datetime.now().strftime("get_ipython().run_line_magic("Y%m%d-%H%M%S"))", "")

tensorboard_callback = keras.callbacks.TensorBoard(logdir, histogram_freq = 1)   


training_history = model.fit(x_train, y_train,
                            validation_split = 0.2,
                            epochs = 500,
                            batch_size = 100,
                            callbacks = [tensorboard_callback])


get_ipython().run_line_magic("load_ext", " tensorboard")


get_ipython().run_line_magic("tensorboard", " --logdir \"./logs/\" --port 7070")


model.evaluate(x_test, y_test)


y_pred = model.predict(x_test)

r2_score(y_test, y_pred)

















































































