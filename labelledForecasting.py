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


youden_folds = []

for _ in range(20):
    random.shuffle(files)


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
    test_input_sequence = []
    test_output_sequence = []
    test_hive_all_data = []


    steps=4
    future_steps = 1
    max_error = 0.01
    count = 0
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
                error = data.at[i+steps+j,'arima_anomaly']
            except:
                print(data.columns)

            # add sample to training/testing depending on whether it is an anomaly
            if error<max_error and count<80:
                train_input_sequence.append(temp_input)
                train_output_sequence.append(temp_output)
            elif count>=80:
                if error<max_error:
                    normal_test_input_sequence.append(temp_input)
                    normal_test_output_sequence.append(temp_output)
                else:
                    anomaly_test_input_sequence.append(temp_input)
                    anomaly_test_output_sequence.append(temp_output)
                test_input_sequence.append(temp_input)
                test_output_sequence.append(temp_output)
                test_hive_all_data.append([entire_sequence])
        hive_testing_indices.append((current_index, len(test_input_sequence)))


    train_input_sequence = np.array(train_input_sequence)
    train_output_sequence = np.array(train_output_sequence)
    original_train_input = np.copy(train_input_sequence)
    normal_test_input_sequence = np.array(normal_test_input_sequence)
    normal_test_output_sequence = np.array(normal_test_output_sequence)
    anomaly_test_input_sequence = np.array(anomaly_test_input_sequence)
    anomaly_test_output_sequence = np.array(anomaly_test_output_sequence)
    original_normal_test_input = np.copy(normal_test_input_sequence)
    original_anomaly_test_input = np.copy(anomaly_test_input_sequence)
    test_input_sequence = np.array(test_input_sequence)
    test_output_sequence = np.array(test_output_sequence)
    original_test_input = np.copy(test_input_sequence)



    # normalise based on training data, but apply to all data
    for i in range(train_input_sequence.shape[2]):
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

    print('model')
    # define model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(200, input_shape=(train_input_sequence.shape[1],train_input_sequence.shape[2]), return_sequences=False))
    #model.add(tf.keras.layers.LSTM(200, return_sequences=False))
    model.add(tf.keras.layers.Dense(200, activation='relu'))
    model.add(tf.keras.layers.Dense(train_output_sequence.shape[1]))
    model.compile(loss='mse', optimizer='adam')
    num_epochs = 100
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

    youden_folds.append(youden)
    continue

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



    # looking at tpr and fpr for different hives
    rec_range = [35.9,32.2]
    sensitivities = []
    oneMinusSpecificities = []
    totalPos = []
    for element in hive_testing_indices:
        if element[0] == element[1]:
            continue
        prediction = model.predict(test_input_sequence[element[0]: element[1]])
        prediction = prediction * (output_max - output_min) + output_min
        observation = test_output_sequence[element[0]: element[1]]

        threshold = 0.25
        true_pos = 0
        false_pos = 0
        entire_pos = 0
        entire_neg = 0
        true_neg = 0
        for index in range(observation.shape[0]):
            # check current temp/humid is not already outside range
            if original_test_input[index][-1][1] > max(rec_range) or original_test_input[index][-1][1] < min(rec_range):
                continue
            next_readings = []
            for j in range(future_steps):
                next_time = test_hive_all_data[index][0][1] + steps + j + 1
                next_readings.append(test_hive_all_data[index][0][0][next_time])

            if min(observation[index]) > max(rec_range) or max(observation[index]) < min(rec_range):
                entire_pos += 1
                if max(prediction[index]) > (max(rec_range) - threshold) or min(prediction[index]) < (
                        min(rec_range) + threshold):
                    # true positive
                    true_pos += 1
                else:
                    # false negative
                    if observation[index][0] > max(rec_range) or observation[index][0] < min(rec_range):
                        pass
            else:
                entire_neg += 1

            if max(prediction[index]) > (max(rec_range) - threshold) or min(prediction[index]) < (
                    min(rec_range) + threshold):
                if min(observation[index]) > max(rec_range) or max(observation[index]) < min(rec_range):
                    pass
                else:
                    # false positive
                    false_pos += 1
                    if max(next_readings) > max(rec_range) or min(next_readings) < min(rec_range):
                        pass
            if max(observation[index]) > max(rec_range) or min(observation[index]) < min(rec_range) or max(
                    prediction[index]) > max(rec_range) or min(prediction[index]) < min(rec_range):
                pass
            else:
                # true negative
                pass

        if entire_pos ==0 or entire_neg==0:
            continue
        totalPos.append(entire_pos/(entire_pos+entire_neg))
        sensitivities.append(true_pos / entire_pos)
        oneMinusSpecificities.append(false_pos / entire_neg)

    fig, ax = plt.subplots()
    scatter = ax.scatter(oneMinusSpecificities, sensitivities, c=totalPos)
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="lower right", title="Proportion positives")
    ax.add_artist(legend1)

    plt.title('TPR vs FPR for each hive, with e=0.25')
    plt.xlabel('1 - specificity')
    plt.ylabel('sensitivity')
    plt.show()

    #exit()

    prediction = model.predict(test_input_sequence)
    prediction = prediction*(output_max-output_min)+output_min
    observation = test_output_sequence


    # classification problem: outside recommended range
    results = [[0,0],[0,0]]
    missed = 0
    next = 0
    rec_range = [35.9,32.2]



    # creating ROC curve
    sensitivities = []
    oneMinusSpecificities = []
    thresholds = []
    threshold = -10

    while threshold<10:
        threshold +=0.1
        true_pos = 0
        false_pos = 0
        entire_pos = 0
        entire_neg = 0
        for index in range(observation.shape[0]):
            # check current temp/humid is not already outside range
            if original_test_input[index][-1][1] > max(rec_range) or original_test_input[index][-1][1]< min(rec_range):
                continue
            next_readings = []
            for j in range(future_steps):
                next_time = test_hive_all_data[index][0][1] + steps + j + 1
                next_readings.append(test_hive_all_data[index][0][0][next_time])

            if min(observation[index])>max(rec_range) or max(observation[index])<min(rec_range):
                entire_pos +=1
                if max(prediction[index]) > (max(rec_range)-threshold) or min(prediction[index]) < (min(rec_range)+threshold):
                    # true positive
                    true_pos +=1
                    results[0][0]+=1
                else:
                    # false negative
                    results[0][1]+=1
                    if observation[index][0]>max(rec_range) or observation[index][0]<min(rec_range):
                        missed +=1
            else:
                entire_neg +=1

            if max(prediction[index])>(max(rec_range)-threshold) or min(prediction[index])<(min(rec_range)+threshold):
                if min(observation[index]) > max(rec_range) or max(observation[index]) < min(rec_range):
                    pass
                else:
                    # false positive
                    false_pos +=1
                    results[1][0]+=1
                    if max(next_readings) >max(rec_range) or min(next_readings)<min(rec_range):
                        next+=1
            if max(observation[index]) > max(rec_range) or min(observation[index]) < min(rec_range) or max(prediction[index])>max(rec_range) or min(prediction[index])<min(rec_range):
                pass
            else:
                # true negative
                results[1][1]+=1

        sensitivities.append(true_pos/entire_pos)
        oneMinusSpecificities.append(false_pos/entire_neg)
        thresholds.append(threshold)

    plt.scatter(oneMinusSpecificities, sensitivities)


    # creating ROC curve
    prediction = list([v] for v in original_test_input[:,-1,-1])
    prediction = np.array(prediction)
    sensitivities = []
    oneMinusSpecificities = []
    thresholds = []
    threshold = -15
    while threshold<10:
        threshold +=0.1
        true_pos = 0
        false_pos = 0
        entire_pos = 0
        entire_neg = 0
        for index in range(observation.shape[0]):
            # check current temp/humid is not already outside range
            if original_test_input[index][-1][1] > max(rec_range) or original_test_input[index][-1][1]< min(rec_range):
                continue
            next_readings = []
            for j in range(future_steps):
                next_time = test_hive_all_data[index][0][1] + steps + j + 1
                next_readings.append(test_hive_all_data[index][0][0][next_time])

            if max(observation[index])>max(rec_range) or min(observation[index])<min(rec_range):
                entire_pos +=1
                if max(prediction[index]) > (max(rec_range)-threshold) or min(prediction[index]) < (min(rec_range)+threshold):
                    # true positive
                    true_pos +=1
                    results[0][0]+=1
                else:
                    # false negative
                    results[0][1]+=1
                    if observation[index][0]>max(rec_range) or observation[index][0]<min(rec_range):
                        missed +=1
            else:
                entire_neg +=1

            if max(prediction[index])>(max(rec_range)-threshold) or min(prediction[index])<(min(rec_range)+threshold):
                if max(observation[index]) > max(rec_range) or min(observation[index]) < min(rec_range):
                    pass
                else:
                    # false positive
                    false_pos +=1
                    results[1][0]+=1
                    if max(next_readings) >max(rec_range) or min(next_readings)<min(rec_range):
                        next+=1
            if max(observation[index]) > max(rec_range) or min(observation[index]) < min(rec_range) or max(prediction[index])>max(rec_range) or min(prediction[index])<min(rec_range):
                pass
            else:
                # true negative
                results[1][1]+=1

        sensitivities.append(true_pos/entire_pos)
        oneMinusSpecificities.append(false_pos/entire_neg)
        thresholds.append(threshold)

    plt.scatter(oneMinusSpecificities, sensitivities)
    plt.ylabel('Sensitivity')
    plt.xlabel('1 - specificity')
    plt.title('ROC curve for predicting internal temp outside range')
    plt.legend(['forecast', 'current temp'])
    plt.plot([0,1],[0,1])
    plt.show()



print(youden_folds)
print(np.mean(youden_folds))
print(np.std(youden_folds))
