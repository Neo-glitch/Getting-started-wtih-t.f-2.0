from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


df = pd.read_csv("./datasets/heart.csv")


# aim is to predict target( 0 no heart disease and 1 = has heart disease)
df.head()


df.isna().sum()  # no na values found


df.describe().T


df.sex.value_counts()


df.cp.value_counts()


# data viz(forrecords by gender and find one with heart disease and not with heart disease)
sns.countplot("sex", hue = "target", data = df)

plt.title("Heart disease frequency for gender")
plt.legend(["No disease", "Yes disease"])

plt.xlabel("Gender(0 = female, 1 = male)")
plt.ylabel("Frequency")

plt.show()


# data viz
plt.figure(figsize = (20, 8))
sns.countplot("age", hue = "target", data = df)

plt.title("Heart disease frequency for Age")
plt.legend(["No disease", "Yes disease"])

plt.xlabel("Age")
plt.ylabel("Frequency")

plt.show()


features = df.drop("target", axis = 1)
target = df[["target"]]


target.sample(10)


# take plain numeric features and standardize 
numeric_features = features[["age", "trestbps", "chol", "thalach", "oldpeak"]].copy()

numeric_features.head()


standardScaler = StandardScaler()

numeric_features = pd.DataFrame(standardScaler.fit_transform(numeric_features),
                               columns = numeric_features.columns,
                               index=numeric_features.index)

numeric_features.describe()


categorical_features = features[["sex", "fbs", "exang", "cp", "ca", "slope", "thal", "restecg"]].copy()

categorical_features.head()


processed_features = pd.concat([numeric_features, categorical_features], axis = 1, sort = False)

processed_features.sample(5)


x_train, x_test, y_train, y_test = train_test_split(processed_features, target,
                                                   test_size = 0.2, random_state = 1)


# split training data further into validaion and train data
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                 test_size = 0.15,
                                                 random_state = 10)


def build_model():
    
    # Bianry clf model
    inputs = tf.keras.Input(shape = (x_train.shape[1],))
    
    dense_layer1 = layers.Dense(12, activation = "relu")
    x = dense_layer1(inputs)  # layer takes input layer as input
    
    dropout_layer = layers.Dropout(0.3)
    x = dropout_layer(x)
    
    dense_layer2 = layers.Dense(8, activation = "relu")
    x = dense_layer2(x)
    
    predictions_layer = layers.Dense(1, activation = "sigmoid")
    predictions = predictions_layer(x)
    
    model = tf.keras.Model(inputs= inputs, outputs = predictions)
    
    model.summary()
    
    model.compile(optimizer = tf.keras.optimizers.Adam(0.001),
                 loss = tf.keras.losses.BinaryCrossentropy(),
                 metrics = ["accuracy",
                           tf.keras.metrics.Precision(0.5),
                           tf.keras.metrics.Recall(0.5)])
    
    return model
    


model = build_model()


# builds pipeline to transform data and feed to model
dataset_train = tf.data.Dataset.from_tensor_slices((x_train.values, y_train.values))

dataset_train = dataset_train.batch(16)

dataset_train.shuffle(128)


num_epochs = 100


# pipeline for validation data
dataset_val = tf.data.Dataset.from_tensor_slices((x_val.values, y_val.values))
dataset_val = dataset_val.batch(16)


model = build_model()

training_history = model.fit(dataset_train, epochs = num_epochs, validation_data = dataset_val)


training_history.history.keys()


# viz model performance
train_acc = training_history.history["accuracy"]
train_loss = training_history.history["loss"]

precision = training_history.history["precision_1"]
recall = training_history.history["recall_1"]

epochs_range = range(num_epochs)

plt.figure(figsize = (10, 10))

plt.subplot(1, 2, 1)

plt.plot(epochs_range, train_acc, label = "Training Accuracy")
plt.plot(epochs_range, train_loss, label = "Training loss")

plt.title("Accuracy and Loss")
plt.legend()

plt.subplot(1, 2, 2)

plt.plot(epochs_range, precision, label = "precision")
plt.plot(epochs_range, recall, label = "recall")

plt.title("Precision and Recall")
plt.legend()


# eval the model
score = model.evaluate(x_test, y_test)

scores_df = pd.Series(score, index = model.metrics_names)

scores_df


y_pred = model.predict(x_test)

y_pred[:10]


# implementing thresholding on score i.e score < 0.5 = 0 and > 0.5 = 1
y_pred = np.where(y_pred <0.5, 0, y_pred)

y_pred = np.where(y_pred >=0.5, 1, y_pred)


y_pred[:10]


pred_results = pd.DataFrame({"y_test": y_test.values.flatten(),
                           "y_pred": y_pred.flatten().astype("int32")}, index = range(len(y_pred)))


pred_results.sample(10)


# confusion matrix
pd.crosstab(pred_results.y_pred, pred_results.y_test)



















































