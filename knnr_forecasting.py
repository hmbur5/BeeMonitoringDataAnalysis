import csv
import numpy as np
import sklearn
print(sklearn.__version__)
import matplotlib.pyplot as plt
import random
from datetime import datetime
import pandas as pd
import os
import math
from sklearn.metrics import r2_score
import pickle
from sklearn.neighbors import KNeighborsRegressor


# get all csv files
directory = 'activity_temp_data/'
files = []
for filename in os.listdir(directory+'labelled/'):
    if filename.endswith(".csv"):
        files.append(filename)
    else:
        continue

random.shuffle(files)


youden_folds = []
tpr_folds = []
fpr_folds = []

anomaly_positives = []
normal_positives = []
anomaly_errors = []
normal_errors = []
hive_data = {}

folds = 60
for hive_index in range(folds):#range(len(files)):
    print(hive_index)


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
                temp_input.append(data['temp'].values[i + j])
                #temp_input.append(data['time'][i + j].hour)
                #temp_input.append([data['time'][i + j].hour,data['temp'].values[i + j]])
                entire_sequence = [list(data['temp'].values), i]
            temp_output = []
            error_vals = [0]
            for j in range(future_steps):
                temp_output.append(data['temp'].values[i+steps+j])
                try:
                    error_vals.append(data.at[i+steps+j,'arima_anomaly'])
                except:
                    pass
            temp_input.append(data['time'][i +steps+ future_steps].hour)
            #if data['time'][i +steps+ j].hour!=12:
            #    continue

            error = np.max(error_vals)

            # add sample to training/testing depending on whether it is an anomaly
            # for the first 80 hives (we withhold 20 hives for testing at the end) we use the normal data from the first
            # 80% of samples for training the model, and the anomaly data as well as anything from the second 20% of
            # samples for finding the best threshold.
            if error<max_error and i<0.8*data.shape[0] and count%folds!=hive_index:
                train_input_sequence.append(temp_input)
                train_output_sequence.append(temp_output)
            elif count%folds!=hive_index and data['time'][i+steps+j].hour!=0:
                if error<max_error:
                    normal_train_input_sequence.append(temp_input)
                    normal_train_output_sequence.append(temp_output)
                else:
                    anomaly_train_input_sequence.append(temp_input)
                    anomaly_train_output_sequence.append(temp_output)
                test_input_sequence.append(temp_input)
                test_output_sequence.append(temp_output)
                test_hive_all_data.append([entire_sequence])
            elif count%folds==hive_index and data['time'][i+steps+j].hour!=0:
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


    # if there are no anomalies in this hive, make sure the list is in the correct shape
    if anomaly_test_input_sequence == []:
        anomaly_test_input_sequence = np.empty([0,8,2])
        anomaly_test_output_sequence = np.empty([0,future_steps])

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
    for i in range(train_input_sequence.shape[1]):
        normal_train_input_sequence[:, i] = (normal_train_input_sequence[:, i] - np.min(
            train_input_sequence[:, i])) / \
                                              (np.max(train_input_sequence[:, i]) - np.min(
                                                  train_input_sequence[:, i]))
        anomaly_train_input_sequence[:, i] = (anomaly_train_input_sequence[:, i] - np.min(
            train_input_sequence[:, i])) / \
                                               (np.max(train_input_sequence[:, i]) - np.min(
                                                   train_input_sequence[:, i]))
        normal_test_input_sequence[:, i] = (normal_test_input_sequence[:, i] - np.min(train_input_sequence[:, i])) / \
                                       (np.max(train_input_sequence[:, i]) - np.min(train_input_sequence[:, i]))
        anomaly_test_input_sequence[:, i] = (anomaly_test_input_sequence[:, i] - np.min(train_input_sequence[:, i])) / \
                                       (np.max(train_input_sequence[:, i]) - np.min(train_input_sequence[:, i]))
        test_input_sequence[:, i] = (test_input_sequence[:, i] - np.min(train_input_sequence[:, i])) / \
                                        (np.max(train_input_sequence[:, i]) - np.min(train_input_sequence[:, i]))
        train_input_sequence[:, i] = (train_input_sequence[:, i] - np.min(train_input_sequence[:, i]))/\
                                        (np.max(train_input_sequence[:, i])-np.min(train_input_sequence[:, i]))



    print('model')
    # define model
    print(np.percentile(normal_train_output_sequence, [25, 50, 75]))
    print(np.percentile(anomaly_train_output_sequence, [25, 50, 75]))

    knnr = KNeighborsRegressor(n_neighbors=10, weights='distance')
    knnr.fit(train_input_sequence, train_output_sequence)
    test_prediction = knnr.predict(test_input_sequence)
    test_error = test_prediction - test_output_sequence
    print(np.mean(np.abs(test_error)))

    print(np.sum(anomaly_train_input_sequence[:,-1]==1))
    print(np.sum(anomaly_train_input_sequence[:, -1] == 0))
    print(len(anomaly_train_input_sequence))


    normal_prediction = knnr.predict(normal_train_input_sequence)
    normal_prediction = normal_prediction
    normal_observation = normal_train_output_sequence
    normal_error = np.abs(normal_prediction - normal_observation)
    normal_error = list(np.max(v) for v in normal_error)

    anomaly_prediction = knnr.predict(anomaly_train_input_sequence)
    anomaly_prediction = anomaly_prediction
    anomaly_observation = anomaly_train_output_sequence
    anomaly_error = np.abs(anomaly_prediction - anomaly_observation)
    anomaly_error = list(np.max(v) for v in anomaly_error)


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
        if false_pos/len(normal_error) <0.5:
            best_thresh = error_thresh
        error_thresh-=0.05
    print(best_thresh)
    print()




    normal_prediction = knnr.predict(normal_test_input_sequence)
    normal_prediction = normal_prediction
    normal_observation = normal_test_output_sequence
    normal_error = np.abs(normal_prediction - normal_observation)
    normal_error = list(np.max(v) for v in normal_error)

    # use try in case there are no anomalies in hive
    try:
        anomaly_prediction = knnr.predict(anomaly_test_input_sequence)
        anomaly_prediction = anomaly_prediction
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


    anomaly_positives += list(np.array(anomaly_error)>best_thresh)
    normal_positives += list(np.array(normal_error)>best_thresh)
    anomaly_errors += list(anomaly_error)
    normal_errors += list(normal_error)
    hive_data[hive_index] = {}
    hive_data[hive_index]['anomaly_error'] = anomaly_error.copy()
    hive_data[hive_index]['normal_error'] = normal_error.copy()
    hive_data[hive_index]['anomaly_positive'] = list(np.array(anomaly_error)>best_thresh).copy()
    hive_data[hive_index]['normal_positive'] = list(np.array(normal_error)>best_thresh).copy()




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

plt.plot([0, 1], [0, 1])
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
threshold = 15
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
