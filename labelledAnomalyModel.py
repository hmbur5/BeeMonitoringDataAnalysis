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
directory = 'activity_temp_data/'
files = []
for filename in os.listdir(directory+'labelled/'):
    if filename.endswith(".csv"):
        files.append(filename)
    else:
        continue


all_predictions = []
all_observations = []
all_detections = []
hive_data = {}

folds = 60
for hive_index in range(folds):#range(len(files)):
    print(hive_index)
    #random.shuffle(files)


    # creating input list that contain 4 consecutive data points of the internal and external temperature
    # and output list being the subsequent internal temperature
    # to avoid using future data in training and past data in testing (which could mean in training it sees the data we're
    # trying to forecast in testing) we split it as we iterate through the beehives.
    train_input_sequence = []
    train_output_sequence = []
    validate_input_sequence = []
    validate_output_sequence = []
    fit_input_sequence = []
    fit_output_sequence = []
    test_input_sequence = []
    test_output_sequence = []
    test_hive_all_data = []

    steps=4
    future_steps = 1
    max_error = 0.01
    count = -1
    hive_testing_indices = []
    for file in files:
        count +=1
        current_index = len(test_input_sequence)
        data = pd.read_csv(directory+'labelled/'+file)
        print(data.columns)
        if 'tempC' not in data.columns:
            continue
        data['time'] = pd.to_datetime(data['time'])
        for i in range(0, data.shape[0]-steps - future_steps, future_steps):
            activity_input = []
            temp_input = []
            for j in range(steps):
                #temp_input.append([data['time'][i + j].hour,data['temp'].values[i + j], data['internal_humidity'].values[i + j]])
                temp_input.append([data['time'][i + j].hour, data['temp'].values[i + j]])
                #temp_input.append([data['time'][i + j].hour, data['internal_humidity'].values[i + j]])
                entire_sequence = [list(data['temp'].values), i]
            temp_output = []
            for j in range(future_steps):
                temp_output.append(data['arima_anomaly'].values[i + steps])
                #temp_output.append(data['temp'].values[i+steps+j])

            try:
                error_vals = [0]
                for j in range(future_steps):
                    error_vals.append(data.at[i + steps + j, 'arima_anomaly'])
                error = np.max(error_vals)
                error = 0
            except:
                print(data.columns)

            # add sample to training/testing depending on hive number
            if count%folds!=hive_index:
                train_input_sequence.append(temp_input)
                train_output_sequence.append(temp_output)
            elif i<0.8*data.shape[0]:
                fit_input_sequence.append(temp_input)
                fit_output_sequence.append(temp_output)
            else:
                test_input_sequence.append(temp_input)
                test_output_sequence.append(temp_output)
                test_hive_all_data.append([entire_sequence])

        hive_testing_indices.append((current_index, len(test_input_sequence)))


    # if there are no anomalies in this hive, make sure the list is in the correct shape


    train_input_sequence = np.array(train_input_sequence)
    train_output_sequence = np.array(train_output_sequence)
    original_train_input = np.copy(train_input_sequence)
    fit_input_sequence = np.array(fit_input_sequence)
    fit_output_sequence = np.array(fit_output_sequence)
    original_fit_input = np.copy(fit_input_sequence)
    test_input_sequence = np.array(test_input_sequence)
    test_output_sequence = np.array(test_output_sequence)
    original_test_input = np.copy(test_input_sequence)


    # normalise based on training data, but apply to all data
    for i in range(train_input_sequence.shape[2]):

        test_input_sequence[:, :, i] = (test_input_sequence[:, :, i] - np.min(train_input_sequence[:, :, i])) / \
                                        (np.max(train_input_sequence[:, :, i]) - np.min(train_input_sequence[:, :, i]))
        fit_input_sequence[:, :, i] = (fit_input_sequence[:, :, i] - np.min(train_input_sequence[:, :, i])) / \
                                       (np.max(train_input_sequence[:, :, i]) - np.min(train_input_sequence[:, :, i]))
        train_input_sequence[:, :, i] = (train_input_sequence[:, :, i] - np.min(train_input_sequence[:, :, i]))/\
                                        (np.max(train_input_sequence[:, :, i])-np.min(train_input_sequence[:, :, i]))
    output_min = np.min(train_output_sequence)
    output_max = np.max(train_output_sequence)


    train_output_sequence = (train_output_sequence - np.min(train_output_sequence))/\
           (np.max(train_output_sequence)-np.min(train_output_sequence))
    fit_output_sequence = (fit_output_sequence - np.min(train_output_sequence))/\
           (np.max(train_output_sequence)-np.min(train_output_sequence))
    test_output_sequence = (test_output_sequence - np.min(train_output_sequence))/\
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
    # add weights as only 2% of data is anomalies
    weights = np.array([[element[0]*49+1] for element in train_output_sequence])


    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(200, input_shape=(train_input_sequence.shape[1],train_input_sequence.shape[2]), return_sequences=False))
    #model.add(tf.keras.layers.LSTM(200, return_sequences=False))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(train_output_sequence.shape[1]))
    model.compile(loss='mse', optimizer='adam')
    num_epochs = 100
    history = model.fit(train_input_sequence, train_output_sequence, epochs=num_epochs,batch_size=128,
                        validation_data=(validation_input_sequence, validation_output_sequence), verbose=0,
                        sample_weight=weights)

    prediction = model.predict(fit_input_sequence)
    prediction = prediction*(output_max-output_min)+output_min
    observation = fit_output_sequence

    best_youden = 0
    threshold = max(prediction)[0]
    print(threshold)
    best_thresh = threshold
    while threshold > 0:
        tp = np.sum((prediction > threshold) & (observation == 1))
        fp = np.sum((prediction > threshold) & (observation == 0))
        tot_pos = np.sum(observation)
        tot_neg = len(observation) - np.sum(observation)
        youden = tp/tot_pos - fp/tot_neg
        if youden > best_youden:
            best_youden = youden
            best_thresh = threshold
        threshold -= 0.05

    prediction = model.predict(train_input_sequence)
    prediction = prediction*(output_max-output_min)+output_min
    observation = train_output_sequence
    all_observations += list(observation)
    all_predictions += list(prediction)
    all_detections += list(prediction>best_thresh)




all_predictions = np.array(all_predictions)
all_observations = np.array(all_observations)
all_detections = np.array(all_detections)
tp = np.sum(np.logical_and(all_detections==1, all_observations==1))
fp = np.sum(np.logical_and(all_detections==1, all_observations==0))
tot_pos = np.sum(all_observations==1)
tot_neg = len(all_observations) - np.sum(all_observations==1)
print(len(all_detections))
print(len(all_observations))
print('tpr')
print(tp)
print(tot_pos)
print(tp/tot_pos)
print('fpr')
print(fp)
print(tot_neg)
print(fp/tot_neg)
print('youden')
print(tp/tot_pos - fp/tot_neg)

#print(all_predictions)
tprs = []
fprs = []
threshold = max(all_predictions)
while threshold>0:
    #print(threshold)
    tp = np.sum(np.logical_and(all_predictions>threshold,all_observations==1))
    fp = np.sum(np.logical_and(all_predictions>threshold, all_observations==0))
    tot_pos = np.sum(all_observations)
    tot_neg = len(all_observations) - np.sum(all_observations)
    threshold-= 0.05
    tprs.append(tp/tot_pos)
    fprs.append(fp/tot_neg)

plt.plot(fprs, tprs)
plt.plot([0, 1], [0, 1])
plt.ylabel('Sensitivity')
plt.xlabel('1 - specificity')
plt.title('ROC curve for identifying anomalies')
plt.show()