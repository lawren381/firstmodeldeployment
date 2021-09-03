from random import seed
from re import X
import tensorflow as tf
import pandas as pd
import numpy as np
import keras
from conda import history
from keras.distribute.distribute_strategy_test import get_model
from keras.models import Sequential, load_model
from keras.layers import *
import scipy
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from numpy import loadtxt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RepeatedKFold
import pickle
from sklearn.preprocessing import StandardScaler
import joblib
from keras.models import load_model
from keras import model_from_json

tf.keras.models.save_model
from sklearn.model_selection import KFold

datar = pd.read_csv("heart.csv")
x = datar.drop["target"].values
y = datar["target"].values
print(x.shape)

# splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# the model
model = Sequential()
model.add(Dense(input_dim=13, activation="relu", units=128, kernel_initializer="uniform"))
model.add(Dense(activation="relu", units=64, kernel_initializer="uniform"))
model.add(Dense(activation="relu", units=64, kernel_initializer="uniform"))
model.add(Dense(activation="relu", units=32, kernel_initializer="uniform"))
model.add(Dense(activation="relu", units=32, kernel_initializer="uniform"))
model.add(Dropout(rate=0.5))
model.add(Dense(activation="relu", units=12, kernel_initializer="uniform"))
model.add(Dense(1, activation="sigmoid", kernel_initializer="uniform"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=10, epochs=150, verbose=0, validation_split=0.05)
print(model.summary())

# Performing prediction and rescaling
y_pred = model.predict(x_test)

# Saving the model for Future Inferences


model_json = model.to_json()
with open("model_train.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_train.h5")

# opening and store file in a variable

json_file = open('model_train.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

# use Keras model_from_json to make a loaded model

loaded_model = tf.keras.models.model_from_json(loaded_model_json)

# load weights into new model

loaded_model.load_weights("model_train.h5")
print("Loaded Model from disk")

# compile and evaluate loaded model

loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

result = loaded_model.predict(x_test)
print(result)
