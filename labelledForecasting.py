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



folds = 60
anomaly_positives = []
normal_positives = []
anomaly_errors = []
normal_errors = []
temp_predictions = []
temp_observations = []
hive_data = {}
rec_range = [35.9, 32.2]  # recommended temperature range

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
    normal_train_input_sequence = []
    normal_train_output_sequence = []
    anomaly_train_input_sequence = []
    anomaly_train_output_sequence = []
    normal_test_input_sequence = []
    normal_test_output_sequence = []
    anomaly_test_input_sequence = []
    anomaly_test_output_sequence = []
    test_input_sequence = []
    test_output_sequence = []
    test_hive_all_data = []

    steps=4
    future_steps = 2
    max_error = 0.01
    count = -1
    hive_testing_indices = []
    for file in files:
        count +=1
        current_index = len(test_input_sequence)
        data = pd.read_csv(directory+'labelled/'+file)
        if 'tempC' not in data.columns:
            continue
        data['time'] = pd.to_datetime(data['time'])
        for i in range(0, data.shape[0]-steps - future_steps, future_steps):
            activity_input = []
            temp_input = []
            for j in range(steps):
                temp_input.append([data['time'][i + j].hour,data['temp'].values[i + j]])
                entire_sequence = [list(data['temp'].values), i]
            temp_output = []
            for j in range(future_steps):
                temp_output.append(data['temp'].values[i+steps+j])

            try:
                error_vals = [0]
                for j in range(future_steps):
                    error_vals.append(data.at[i + steps + j, 'arima_anomaly'])
                error = np.max(error_vals)
            except:
                print(data.columns)

            # add sample to training/testing depending on whether it is an anomaly
            # for the first 80 hives (we withhold 20 hives for testing at the end) we use the normal data from the first
            # 80% of samples for training the model, and the anomaly data as well as anything from the second 20% of
            # samples for finding the best threshold.
            if error<max_error and i<0.8*data.shape[0] and count%folds!=hive_index:
                train_input_sequence.append(temp_input)
                train_output_sequence.append(temp_output)
            elif count % folds != hive_index and data['time'][i + steps + j].hour != 0:
                if error<max_error:
                    normal_train_input_sequence.append(temp_input)
                    normal_train_output_sequence.append(temp_output)
                else:
                    anomaly_train_input_sequence.append(temp_input)
                    anomaly_train_output_sequence.append(temp_output)
                # for looking at rec range predictions, only use data whose current time step is within range
                if data['temp'].values[i+steps-1]>max(rec_range) or data['temp'].values[i+steps-1]<min(rec_range):
                    test_input_sequence.append(temp_input)
                    test_output_sequence.append(temp_output)
            elif count % folds == hive_index and data['time'][i + steps + j].hour != 0:
                if error<max_error:
                    normal_test_input_sequence.append(temp_input)
                    normal_test_output_sequence.append(temp_output)
                else:
                    anomaly_test_input_sequence.append(temp_input)
                    anomaly_test_output_sequence.append(temp_output)
                # for looking at rec range predictions, only use data whose current time step is within range
                if data['temp'].values[i+steps-1]>max(rec_range) or data['temp'].values[i+steps-1]<min(rec_range):
                    test_input_sequence.append(temp_input)
                    test_output_sequence.append(temp_output)
        hive_testing_indices.append((current_index, len(test_input_sequence)))


    # if there are no normals/anomalies in this hive, make sure the list is in the correct shape
    if anomaly_test_input_sequence == []:
        anomaly_test_input_sequence = np.empty([0,4,2])
        anomaly_test_output_sequence = np.empty([0,1])
    if normal_test_input_sequence == []:
        normal_test_input_sequence = np.empty([0,4,2])
        normal_test_output_sequence = np.empty([0,1])

    train_input_sequence = np.array(train_input_sequence)
    train_output_sequence = np.array(train_output_sequence)
    original_train_input = np.copy(train_input_sequence)
    normal_train_input_sequence = np.array(normal_train_input_sequence)
    normal_train_output_sequence = np.array(normal_train_output_sequence)
    anomaly_train_input_sequence = np.array(anomaly_train_input_sequence)
    anomaly_train_output_sequence = np.array(anomaly_train_output_sequence)
    normal_test_input_sequence = np.array(normal_test_input_sequence)
    normal_test_output_sequence = np.array(normal_test_output_sequence)
    anomaly_test_input_sequence = np.array(anomaly_test_input_sequence)
    anomaly_test_output_sequence = np.array(anomaly_test_output_sequence)
    original_normal_train_input = np.copy(normal_train_input_sequence)
    original_anomaly_train_input = np.copy(anomaly_train_input_sequence)
    original_normal_test_input = np.copy(normal_test_input_sequence)
    original_anomaly_test_input = np.copy(anomaly_test_input_sequence)
    test_input_sequence = np.array(test_input_sequence)
    test_output_sequence = np.array(test_output_sequence)
    original_test_input = np.copy(test_input_sequence)


    # normalise based on training data, but apply to all data
    for i in range(train_input_sequence.shape[2]):
        normal_train_input_sequence[:, :, i] = (normal_train_input_sequence[:, :, i] - np.min(
            train_input_sequence[:, :, i])) / \
                                              (np.max(train_input_sequence[:, :, i]) - np.min(
                                                  train_input_sequence[:, :, i]))
        anomaly_train_input_sequence[:, :, i] = (anomaly_train_input_sequence[:, :, i] - np.min(
            train_input_sequence[:, :, i])) / \
                                               (np.max(train_input_sequence[:, :, i]) - np.min(
                                                   train_input_sequence[:, :, i]))
        normal_test_input_sequence[:, :, i] = (normal_test_input_sequence[:, :, i] - np.min(train_input_sequence[:, :, i])) / \
                                       (np.max(train_input_sequence[:, :, i]) - np.min(train_input_sequence[:, :, i]))
        anomaly_test_input_sequence[:, :, i] = (anomaly_test_input_sequence[:, :, i] - np.min(train_input_sequence[:, :, i])) / \
                                       (np.max(train_input_sequence[:, :, i]) - np.min(train_input_sequence[:, :, i]))
        test_input_sequence[:, :, i] = (test_input_sequence[:, :, i] - np.min(train_input_sequence[:, :, i])) / \
                                        (np.max(train_input_sequence[:, :, i]) - np.min(train_input_sequence[:, :, i]))
        train_input_sequence[:, :, i] = (train_input_sequence[:, :, i] - np.min(train_input_sequence[:, :, i]))/\
                                        (np.max(train_input_sequence[:, :, i])-np.min(train_input_sequence[:, :, i]))
    output_min = np.min(train_output_sequence)
    output_max = np.max(train_output_sequence)


    train_output_sequence = (train_output_sequence - np.min(train_output_sequence))/\
           (np.max(train_output_sequence)-np.min(train_output_sequence))
    normal_train_output_sequence = (normal_train_output_sequence - np.min(train_output_sequence))/\
           (np.max(train_output_sequence)-np.min(train_output_sequence))
    anomaly_train_output_sequence = (anomaly_train_output_sequence - np.min(train_output_sequence))/\
           (np.max(train_output_sequence)-np.min(train_output_sequence))
    normal_test_output_sequence = (normal_test_output_sequence - np.min(train_output_sequence))/\
           (np.max(train_output_sequence)-np.min(train_output_sequence))
    anomaly_test_output_sequence = (anomaly_test_output_sequence - np.min(train_output_sequence))/\
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

    model_fit_normal_errors = []
    model_fit_anomaly_errors = []
    model_test_normal_errors = []
    model_test_anomaly_errors = []
    model_test_temp_predictions = []
    model_test_temp_observations = []
    for model_iter in range(5):
        print('model')
        # define model
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(50, input_shape=(train_input_sequence.shape[1],train_input_sequence.shape[2]), return_sequences=False))
        #model.add(tf.keras.layers.LSTM(200, return_sequences=False))
        #model.add(tf.keras.layers.Dense(10, activation='relu'))
        model.add(tf.keras.layers.Dense(train_output_sequence.shape[1]))
        model.compile(loss='mse', optimizer='adam')
        num_epochs = 100
        history = model.fit(train_input_sequence, train_output_sequence, epochs=num_epochs,batch_size=128,
                            validation_data=(validation_input_sequence, validation_output_sequence), verbose=0)


        normal_prediction = model.predict(normal_train_input_sequence)
        normal_prediction = normal_prediction*(output_max-output_min)+output_min
        normal_observation = normal_train_output_sequence
        normal_error = np.abs(normal_prediction - normal_observation)
        normal_error = list(np.max(v) for v in normal_error)


        anomaly_prediction = model.predict(anomaly_train_input_sequence)
        anomaly_prediction = anomaly_prediction*(output_max-output_min)+output_min
        anomaly_observation = anomaly_train_output_sequence
        anomaly_error = np.abs(anomaly_prediction - anomaly_observation)
        anomaly_error = list(np.max(v) for v in anomaly_error)

        model_fit_anomaly_errors.append(anomaly_error)
        model_fit_normal_errors.append(normal_error)

        try:
            normal_prediction = model.predict(normal_test_input_sequence)
        except ValueError:
            shape_prediction = list(normal_prediction.shape)
            shape_prediction[0] = 0
            normal_prediction = np.empty(shape_prediction)
        normal_prediction = normal_prediction * (output_max - output_min) + output_min
        normal_observation = normal_test_output_sequence
        normal_error = np.abs(normal_prediction - normal_observation)
        normal_error = list(np.max(v) for v in normal_error)

        # use try in case there are no anomalies in hive
        try:
            anomaly_prediction = model.predict(anomaly_test_input_sequence)
            anomaly_prediction = anomaly_prediction * (output_max - output_min) + output_min
            anomaly_observation = anomaly_test_output_sequence
            anomaly_error = np.abs(anomaly_prediction - anomaly_observation)
            anomaly_error = list(np.max(v) for v in anomaly_error)
        except:
            shape_prediction = list(normal_prediction.shape)
            shape_prediction[0] = 0
            anomaly_prediction = np.empty(shape_prediction)
            anomaly_observation = np.empty(shape_prediction)
            anomaly_error = np.empty(shape_prediction)
            anomaly_error = list(np.max(v) for v in anomaly_error)

        model_test_anomaly_errors.append(anomaly_error)
        model_test_normal_errors.append(normal_error)

        # using first time step to look at unsafe temp detection for all data
        all_prediction = model.predict(test_input_sequence)
        all_prediction = all_prediction * (output_max - output_min) + output_min
        all_observation = test_output_sequence
        model_test_temp_predictions.append(list(all_prediction[:,0]))
        model_test_temp_observations.append(list(all_observation[:,0]))

    anomaly_error = np.median(model_fit_anomaly_errors, axis = 0)
    print(len(anomaly_error))
    normal_error = np.median(model_fit_normal_errors, axis=0)
    print(len(normal_error))

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
        #if false_pos/len(normal_error) <0.5:
        #    best_thresh = error_thresh
        error_thresh-=0.05
    print(best_thresh)
    print()


    anomaly_error = np.median(model_test_anomaly_errors, axis = 0)
    normal_error = np.median(model_test_normal_errors, axis=0)

    anomaly_positives += list(np.array(anomaly_error)>best_thresh)
    normal_positives += list(np.array(normal_error)>best_thresh)
    anomaly_errors += list(anomaly_error)
    normal_errors += list(normal_error)
    temp_predictions += list(np.median(model_test_temp_predictions, axis = 0))
    temp_observations += list(np.median(model_test_temp_observations, axis = 0))
    if hive_index not in hive_data.keys():
        hive_data[hive_index] = {}
    hive_data[hive_index] = {}
    hive_data[hive_index]['anomaly_error'] = anomaly_error.copy()
    hive_data[hive_index]['normal_error'] = normal_error.copy()
    hive_data[hive_index]['anomaly_positive'] = list(np.array(anomaly_error)>best_thresh).copy()
    hive_data[hive_index]['normal_positive'] = list(np.array(normal_error)>best_thresh).copy()
    hive_data[hive_index]['temp predictions'] = np.median(model_test_temp_predictions, axis = 0)
    hive_data[hive_index]['temp observations'] = np.median(model_test_temp_observations, axis = 0)

    print(np.mean(np.abs(normal_error)))









print('TPR all hives')
print(np.sum(anomaly_positives)/len(anomaly_positives))
print('FPR all hives')
print(np.sum(normal_positives)/len(normal_positives))
print('youden all hives')
print(np.sum(anomaly_positives)/len(anomaly_positives) - np.sum(normal_positives)/len(normal_positives))


tprs = []
fprs = []
totalPos = []
fig, ax = plt.subplots()
for i in hive_data.keys():
    if len(hive_data[i]['anomaly_positive']) == 0:
        print(i)
        continue
    tprs.append(np.sum(hive_data[i]['anomaly_positive'])/len(hive_data[i]['anomaly_positive']))
    fprs.append(np.sum(hive_data[i]['normal_positive']) / len(hive_data[i]['normal_positive']))
    totalPos.append(len(hive_data[i]['anomaly_positive'])/(len(hive_data[i]['anomaly_positive'])+len(hive_data[i]['normal_positive'])))

scatter = ax.scatter(fprs, tprs, c=totalPos)
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower right", title="Proportion positives")
ax.add_artist(legend1)

plt.title('TPR vs FPR for each hive')
plt.xlabel('1 - specificity')
plt.ylabel('sensitivity')
plt.show()

print('anomaly error average')
print(np.mean(anomaly_errors))
print('normal error average')
print(np.mean(normal_errors))


tprs = []
fprs = []
threshold = max(max(normal_errors), max(anomaly_errors))
while threshold>0:
    tp = 0
    fp = 0
    tot_pos = 0
    tot_neg = 0
    for i in hive_data.keys():
        tp += np.sum(np.array(hive_data[i]['anomaly_error'])>threshold)

        fp += np.sum(np.array(hive_data[i]['normal_error'])>threshold)

        tot_pos += len(hive_data[i]['anomaly_error'])
        tot_neg += len(hive_data[i]['normal_error'])
    threshold-= 0.05
    tprs.append(tp/tot_pos)
    fprs.append(fp/tot_neg)

plt.plot(fprs, tprs)
plt.plot([0, 1], [0, 1])
plt.ylabel('Sensitivity')
plt.xlabel('1 - specificity')
plt.title('ROC curve for identifying anomalies')
plt.show()








# looking at temp predictions


# find best threshold
threshold = -20
best_thresh = None
youden = 0
tprs = [0]
fprs = [0]
while threshold<5:
    tp = 0
    fp = 0
    tot_pos = 0
    tot_neg = 0
    detected_positives = np.logical_or(np.array(temp_predictions) > max(rec_range)+threshold, np.array(temp_predictions) < min(rec_range)-threshold)
    observed_positives = np.logical_or(np.array(temp_observations) > max(rec_range), np.array(temp_observations) < min(rec_range))

    threshold +=0.1
    tp = np.sum(np.logical_and(detected_positives, observed_positives))
    fp = np.sum(np.logical_and(detected_positives, ~observed_positives))
    tot_pos = np.sum(observed_positives)
    tot_neg = np.sum(~observed_positives)
    tprs.append(tp/tot_pos)
    fprs.append(fp/tot_neg)

    if tp/tot_pos - fp/tot_neg > youden and fp/tot_neg>0.5:
        best_thresh = threshold
        youden = tp/tot_pos - fp/tot_neg


plt.plot(fprs, tprs)
plt.plot([0, 1], [0, 1])
plt.ylabel('Sensitivity')
plt.xlabel('1 - specificity')
plt.title('ROC curve for temp outside safe range')
plt.show()




# curve for individual hives
tprs = []
fprs = []
totalPos = []
threshold = best_thresh
for i in hive_data.keys():

    prediction = np.array(hive_data[i]['temp predictions'])
    observation = np.array(hive_data[i]['temp observations'])

    detected_positives = np.logical_or(prediction > max(rec_range) + threshold,
                                       prediction < min(rec_range) - threshold)
    observed_positives = np.logical_or(observation > max(rec_range),
                                       observation < min(rec_range))

    tp = np.sum(np.logical_and(detected_positives, observed_positives))
    fp = np.sum(np.logical_and(detected_positives, ~observed_positives))
    tot_pos = np.sum(observed_positives)
    tot_neg = np.sum(~observed_positives)
    tprs.append(tp / tot_pos)
    fprs.append(fp / tot_neg)
    totalPos.append(np.sum(observed_positives) / len(observed_positives))

fig, ax = plt.subplots()
scatter = ax.scatter(fprs, tprs, c=totalPos)
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower right", title="Proportion positives")
ax.add_artist(legend1)

plt.title('TPR vs FPR for each hive, with e='+str(threshold))
plt.xlabel('1 - specificity')
plt.ylabel('sensitivity')
plt.plot([0,1],[0,1])
plt.show()
