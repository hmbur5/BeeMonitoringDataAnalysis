import csv
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sklearn
print(sklearn.__version__)
import matplotlib.pyplot as plt
import random
from datetime import datetime
import pandas as pd
import os
import math
from keras.callbacks import ModelCheckpoint
import keras.backend as k
from sklearn.metrics import r2_score
import pickle

# get all csv files
variable = 'activity'
directory = 'activity_temp_data/'
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

steps=4
future_steps = 1
for file in files:
    data = pd.read_csv(directory+file)
    if 'tempC' not in data.columns:
        continue
    data['time'] = pd.to_datetime(data['time'])
    for i in range(0, data.shape[0]-steps - future_steps, future_steps):
        input = []
        for j in range(steps):
            weather = data.loc[:,['tempC', 'uvIndex', 'cloudcover', 'visibility', 'humidity', 'windspeedKmph']]
            weatherList = weather.loc[i+j+1].tolist()
            input.append([data['time'][i + j].hour] + weatherList + [data[variable].values[i + j]])
        output = []
        for j in range(future_steps):
            output.append(data[variable].values[i+steps+j])
        train_input_sequence.append(input)
        train_output_sequence.append(output)

train_input_sequence = np.array(train_input_sequence)
train_output_sequence = np.array(train_output_sequence)
original_train_input = np.copy(train_input_sequence)


# normalise based on training data, but apply to all data
scalers = []
for i in range(train_input_sequence.shape[2]):
    scalers.append([np.min(train_input_sequence[:, :, i]), np.max(train_input_sequence[:, :, i])])
    train_input_sequence[:, :, i] = (train_input_sequence[:, :, i] - np.min(train_input_sequence[:, :, i]))/\
                                    (np.max(train_input_sequence[:, :, i])-np.min(train_input_sequence[:, :, i]))
output_min = np.min(train_output_sequence)
output_max = np.max(train_output_sequence)

train_output_sequence = (train_output_sequence - np.min(train_output_sequence))/\
       (np.max(train_output_sequence)-np.min(train_output_sequence))


#dump(scaler2, open('temp_forecast_scaler2.pkl', 'wb'))


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
model.add(tf.keras.layers.LSTM(200, input_shape=(train_input_sequence.shape[1],train_input_sequence.shape[2]), return_sequences=False))
#model.add(tf.keras.layers.LSTM(200, return_sequences=False))
model.add(tf.keras.layers.Dense(200, activation='relu'))
model.add(tf.keras.layers.Dense(train_output_sequence.shape[1]))
model.compile(loss='mse', optimizer='adam')
num_epochs = 50
history = model.fit(train_input_sequence, train_output_sequence, epochs=num_epochs,batch_size=128,
                    validation_data=(validation_input_sequence, validation_output_sequence), verbose=1)


prediction = model.predict(train_input_sequence)
#prediction = prediction*(output_max-output_min)+output_min
observation = train_output_sequence

error = np.abs(prediction-observation)
error = list(v[0] for v in error)
sorted_error = np.sort(error)
max_error = sorted_error[int(0.8*len(error))]



# temp
# get all csv files
variable = 'temp'
directory = 'activity_temp_data/'
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
normal_test_input_sequence = []
normal_test_output_sequence = []
anomaly_test_input_sequence = []
anomaly_test_output_sequence = []
test_hive_all_data = []

for file in files:
    data = pd.read_csv(directory+file)
    if 'tempC' not in data.columns:
        continue
    data['time'] = pd.to_datetime(data['time'])
    for i in range(0, data.shape[0]-steps - future_steps, future_steps):
        activity_input = []
        temp_input = []
        for j in range(steps):
            weather = data.loc[:, ['tempC', 'uvIndex', 'cloudcover', 'visibility', 'humidity', 'windspeedKmph']]
            weatherList = weather.loc[i + j + 1].tolist()
            activity_input.append([data['time'][i + j].hour] + weatherList + [data['activity'].values[i + j]])
            temp_input.append([data['time'][i + j].hour,data['temp'].values[i + j]])
            entire_sequence = [list(data[variable].values), i]
        activity_output = []
        temp_output = []
        for j in range(future_steps):
            activity_output.append(data['activity'].values[i+steps+j])
            temp_output.append(data['temp'].values[i+steps+j])

        # get error of sample from activity model
        activity_input = [activity_input]
        activity_input = np.array(activity_input)

        for k in range(activity_input.shape[2]):
            activity_input[:,:,k] = (activity_input[:,:,k] - scalers[k][0]) / \
                                            (scalers[k][1] - scalers[k][0])
        activity_output = (activity_output - output_min) / \
                                (output_max - output_min)
        prediction = model.predict(activity_input)
        error = prediction[0] - activity_output
        error = np.abs(error[0])


        # add sample to training/testing depending on whether it is an anomaly
        if i < 0.7 * data.shape[0] and error<max_error:
            train_input_sequence.append(temp_input)
            train_output_sequence.append(temp_output)
        else:
            # skip midnight reading in test
            if input[-1][0] == 0:
                continue
            if error<max_error:
                normal_test_input_sequence.append(temp_input)
                normal_test_output_sequence.append(temp_output)
            else:
                anomaly_test_input_sequence.append(temp_input)
                anomaly_test_output_sequence.append(temp_output)
            test_hive_all_data.append([entire_sequence])


train_input_sequence = np.array(train_input_sequence)
train_output_sequence = np.array(train_output_sequence)
original_train_input = np.copy(train_input_sequence)
normal_test_input_sequence = np.array(normal_test_input_sequence)
normal_test_output_sequence = np.array(normal_test_output_sequence)
anomaly_test_input_sequence = np.array(anomaly_test_input_sequence)
anomaly_test_output_sequence = np.array(anomaly_test_output_sequence)


# normalise based on training data, but apply to all data
for i in range(train_input_sequence.shape[2]):
    normal_test_input_sequence[:, :, i] = (normal_test_input_sequence[:, :, i] - np.min(train_input_sequence[:, :, i])) / \
                                   (np.max(train_input_sequence[:, :, i]) - np.min(train_input_sequence[:, :, i]))
    anomaly_test_input_sequence[:, :, i] = (anomaly_test_input_sequence[:, :, i] - np.min(train_input_sequence[:, :, i])) / \
                                   (np.max(train_input_sequence[:, :, i]) - np.min(train_input_sequence[:, :, i]))
    train_input_sequence[:, :, i] = (train_input_sequence[:, :, i] - np.min(train_input_sequence[:, :, i]))/\
                                    (np.max(train_input_sequence[:, :, i])-np.min(train_input_sequence[:, :, i]))
output_min = np.min(train_output_sequence)
output_max = np.max(train_output_sequence)

train_output_sequence = (train_output_sequence - np.min(train_output_sequence))/\
       (np.max(train_output_sequence)-np.min(train_output_sequence))
normal_test_output_sequence = (normal_test_output_sequence - np.min(train_output_sequence))/\
       (np.max(train_output_sequence)-np.min(train_output_sequence))
anomaly_test_output_sequence = (anomaly_test_output_sequence - np.min(train_output_sequence))/\
       (np.max(train_output_sequence)-np.min(train_output_sequence))

#dump(scaler2, open('temp_forecast_scaler2.pkl', 'wb'))



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

print('model')
# define model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(200, input_shape=(train_input_sequence.shape[1],train_input_sequence.shape[2]), return_sequences=False))
#model.add(tf.keras.layers.LSTM(200, return_sequences=False))
model.add(tf.keras.layers.Dense(200, activation='relu'))
model.add(tf.keras.layers.Dense(train_output_sequence.shape[1]))
model.compile(loss='mse', optimizer='adam')
num_epochs = 50
history = model.fit(train_input_sequence, train_output_sequence, epochs=num_epochs,batch_size=128,
                    validation_data=(validation_input_sequence, validation_output_sequence), verbose=1)


normal_prediction = model.predict(normal_test_input_sequence)
normal_prediction = normal_prediction*(output_max-output_min)+output_min
normal_observation = normal_test_output_sequence
normal_error = np.abs(normal_prediction - normal_observation)
normal_error = list(v[0] for v in normal_error)


anomaly_prediction = model.predict(anomaly_test_input_sequence)
anomaly_prediction = anomaly_prediction*(output_max-output_min)+output_min
anomaly_observation = anomaly_test_output_sequence
anomaly_error = np.abs(anomaly_prediction - anomaly_observation)
anomaly_error = list(v[0] for v in anomaly_error)

error_thresh = max([max(anomaly_error), max(normal_error)])
max_diff = 0
errors = []
youdens = []
sensitivity = []
oneMinusSpecificity = []
while error_thresh>0:
    true_pos = sum(anomaly_error>error_thresh)
    false_pos = sum(normal_error>error_thresh)
    sensitivity.append(true_pos/len(anomaly_error))
    oneMinusSpecificity.append(false_pos/len(normal_error))
    youden = true_pos/len(anomaly_error) - false_pos/len(normal_error)
    errors.append(error_thresh)
    youdens.append(youden)
    if youden > max_diff:
        max_diff = youden
        best_thresh = error_thresh
    error_thresh-=0.01

print(best_thresh)
true_pos = sum(anomaly_error>best_thresh)
false_pos = sum(normal_error>best_thresh)
print(true_pos)
print(len(anomaly_error))
print(false_pos)
print(len(normal_error))
print(true_pos/len(anomaly_error))
print(false_pos/len(normal_error))
youden = true_pos/len(anomaly_error) - false_pos/len(normal_error)



plt.plot(errors, youdens)
plt.ylabel('Youden index')
plt.xlabel('Temperature error threshold')
plt.show()

plt.scatter(oneMinusSpecificity, sensitivity)
plt.ylabel('Sensitivity')
plt.xlabel('1 - specificity')
plt.title('ROC curve')
plt.plot([0,1],[0,1])
plt.show()