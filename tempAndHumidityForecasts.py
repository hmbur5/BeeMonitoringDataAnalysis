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


'''df = pd.DataFrame(index = list(range(pickle.load(open("activity_error.obj",'rb')).shape[0])), columns=['activity error', 'temp error'])
df['activity error'] = pickle.load(open("activity_error.obj",'rb'))
df['temp error'] = pickle.load(open("temp_error.obj",'rb'))
#df['humidity error'] = pickle.load(open("humidity_error.obj",'rb'))
df['warning'] = (df['temp error']>5) #& (df['humidity error']>8)
print(np.median(df['temp error']))
print(np.median(df['activity error']))
#print(np.median(df['humidity error']))
print(pd.crosstab(df['activity error'] > 1500, df['warning']))
print('true neg\t false pos\nfalse neg\ttrue pos')
#plt.scatter(df['temp error'], df['activity error'])
exit()'''


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
test_input_sequence = []
test_output_sequence = []
test_hive_all_data = []

steps=4
future_steps = 2
for file in files:
    data = pd.read_csv(directory+file)
    if 'tempC' not in data.columns:
        continue
    data['time'] = pd.to_datetime(data['time'])
    for i in range(0, data.shape[0]-steps - future_steps, future_steps):
        input = []
        for j in range(steps):
            if variable=='activity':
                weather = data.loc[:,['tempC', 'uvIndex', 'cloudcover', 'visibility', 'humidity', 'windspeedKmph']]
                weatherList = weather.loc[i+j+1].tolist()
                #weatherList = []
                input.append([data['time'][i + j].hour] + weatherList + [data[variable].values[i + j]])
            else:
                input.append([data['time'][i+j].hour,data[variable].values[i+j]])
            entire_sequence = [list(data[variable].values), i]
        output = []
        for j in range(future_steps):
            output.append(data[variable].values[i+steps+j])
        # check data is valid for temperature
        if variable!='temp ' or all(j >= 28 for i,j in input):
            if i<0.8*data.shape[0]:
                train_input_sequence.append(input)
                train_output_sequence.append(output)
            else:
                # skip midnight reading in test
                if input[-1][0]==0:
                    continue
                test_input_sequence.append(input)
                test_output_sequence.append(output)
                test_hive_all_data.append([entire_sequence])

train_input_sequence = np.array(train_input_sequence)
train_output_sequence = np.array(train_output_sequence)
test_input_sequence = np.array(test_input_sequence)
test_output_sequence = np.array(test_output_sequence)
original_test_input = np.copy(test_input_sequence)


original_input = np.copy(test_input_sequence)
# normalise based on training data, but apply to all data
for i in range(train_input_sequence.shape[2]):
    print(np.min(train_input_sequence[:, :, i]))
    print(np.max(train_input_sequence[:, :, i]))
    test_input_sequence[:, :, i] = (test_input_sequence[:, :, i] - np.min(train_input_sequence[:, :, i]))/\
                                    (np.max(train_input_sequence[:, :, i])-np.min(train_input_sequence[:, :, i]))
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
prediction = prediction*(output_max-output_min)+output_min
observation = test_output_sequence


#print(np.mean(np.abs(prediction-observation)))
#print(np.mean(np.abs(observation-original_input[:,-1,-1])))
#print(np.mean(np.abs(observation-original_input[:,-4,-1])))

#print('r2')
#print(r2_score(observation, prediction))

filehandler = open(variable+"_error.obj","wb")
pickle.dump(np.abs(observation-prediction),filehandler)
filehandler.close()

#print('previous day')
#print(r2_score(observation,original_input[:,-4,-1]))


'''
print(np.sqrt(np.mean(np.square(observation-prediction))))
print(np.sqrt(np.mean(np.square(observation-original_input[:,-1,1]))))

print(np.mean(np.abs(observation-prediction)))
print(np.mean(np.abs(observation-original_input[:,0,1])))

plt.scatter(prediction, observation, alpha=0.5)
plt.scatter(original_input[:,-1,1], observation, alpha=0.5)
plt.plot([26,35],[26,35])
plt.clf()

plt.hist(observation-prediction, bins=20, density=True)
plt.title('Probability distribution of error between prediction and observation')
plt.clf()'''


# confusion matrix for outside recommended range
results = [[0,0],[0,0]]
missed = 0
next = 0
if variable=='temp':
    rec_range = [35.9,32.2]
elif variable=='humidity':
    rec_range = [53, 61]



# creating ROC curve
sensitivities = []
oneMinusSpecificities = []
thresholds = []
threshold = -10

while threshold<10:
    threshold +=0.01
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
prediction = list([v,v] for v in original_input[:,-1,-1])
prediction = np.array(prediction)
sensitivities = []
oneMinusSpecificities = []
thresholds = []
threshold = -15
while threshold<10:
    threshold +=0.01
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
plt.ylabel('Sensitivity')
plt.xlabel('1 - specificity')
plt.title('ROC curve for predicting internal temp outside range - prolonged exposure over next 2 time-steps')
plt.legend(['forecast', 'current temp'])
plt.plot([0,1],[0,1])
plt.show()



print(results)
print(next)
print(missed)

