import csv
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import random
from datetime import datetime
from pickle import dump
import pandas as pd
import os
import math
from keras.callbacks import ModelCheckpoint
import keras.backend as k



# get all csv files
directory = 'temp_data/'
files = []
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        files.append(filename)
    else:
        continue


# creating input list that contain 4 consecutive data points of the internal and external temperature
# and output list being the subsequent internal temperature
# to avoid using future data in training and past data in testing (which could mean in training it sees the data we're
# trying to forecast in testing) we split it as we iterate through the beehives.
train_input_sequence = []
train_output_sequence = []
validate_input_sequence = []
validate_output_sequence = []
test_input_sequence = []
test_output_sequence = []

steps=4
for file in files:
    temp_data = pd.read_csv(directory+file)
    for i in range(temp_data.shape[0]-steps-1):
        input = []
        for j in range(steps):
            input.append([temp_data.outside_temp[i+j],temp_data.temp[i+j]])
        # check tempetrature is valid data
        if all(j >= 30 for i,j in input):
            if i<0.8*temp_data.shape[0]:
                train_input_sequence.append(input)
                train_output_sequence.append([temp_data.temp[i+steps]])
            else:
                test_input_sequence.append(input)
                test_output_sequence.append([temp_data.temp[i+steps]])

train_input_sequence = np.array(train_input_sequence)
train_output_sequence = np.array(train_output_sequence)
test_input_sequence = np.array(test_input_sequence)
test_output_sequence = np.array(test_output_sequence)


print(train_input_sequence.shape)
print(np.sum(train_input_sequence))
print(train_output_sequence.shape)
print(np.sum(train_output_sequence))

original_input = np.copy(test_input_sequence)
# normalise based on training data, but apply to all data
for i in range(train_input_sequence.shape[2]):
    scaler1 = MinMaxScaler(feature_range=(0, 1))
    scaler1 = scaler1.fit(train_input_sequence[:, :, i])
    train_input_sequence[:, :, i] = scaler1.transform(train_input_sequence[:, :, i])
    test_input_sequence[:, :, i] = scaler1.transform(test_input_sequence[:, :, i])
    dump(scaler1, open('temp_forecast_scaler1'+str(i)+'.pkl','wb'))
scaler2 = MinMaxScaler(feature_range=(0, 1))
scaler2 = scaler2.fit(train_output_sequence)
train_output_sequence = scaler2.transform(train_output_sequence)
dump(scaler2, open('temp_forecast_scaler2.pkl', 'wb'))


# shuffle and split training into training and validation
indices = list(range(train_input_sequence.shape[0]))
random.shuffle(indices)
train_input_sequences = train_input_sequence[indices]
train_output_sequences = train_output_sequence[indices]
training_portion = np.float64(0.8)
train_size = int(train_input_sequence.shape[0] * training_portion)
validation_input_sequence = train_input_sequence[train_size:]
validation_output_sequence = train_output_sequence[train_size:]
train_input_sequence = train_input_sequence[0: train_size]
train_output_sequence = train_output_sequence[0: train_size]


# define model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(64, input_shape=(train_input_sequence.shape[1],train_input_sequence.shape[2]), return_sequences=False))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(train_output_sequence.shape[1]))
model.compile(loss='mse', optimizer='adam')
num_epochs = 100
history = model.fit(train_input_sequence, train_output_sequence, epochs=num_epochs,batch_size=128,
                    validation_data=(validation_input_sequence, validation_output_sequence), verbose=1)


# Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model) # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)

prediction = model.predict(test_input_sequence)
prediction = scaler2.inverse_transform(prediction)
observation = test_output_sequence

print(np.sqrt(np.mean(np.square(observation-prediction))))
print(np.sqrt(np.mean(np.square(observation-original_input[:,-1,1]))))

print(np.mean(np.square(observation-prediction)))
print(np.mean(np.abs(observation-original_input[:,-1,1])))

plt.scatter(prediction, observation, alpha=0.5)
plt.scatter(original_input[:,-1,1], observation, alpha=0.5)
plt.plot([26,35],[26,35])
plt.show()


