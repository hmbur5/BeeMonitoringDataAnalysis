import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta




# get all csv files
directory = 'activity_temp_data/labelled/'
files = []
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        files.append(filename)
    else:
        continue
file = files[0]
#file = 'site1677.0_device1722.0.csv'


steps=4
future_steps = 2
data = pd.read_csv(directory+file)
data.index = pd.to_datetime(data['time'])
data['file'] = file

print(data.columns)

for file in files[1:]:
    data2 = pd.read_csv(directory+file)
    if 'tempC' not in data2.columns:
        continue
    # add some years to each file (for sufficient gap between hives)
    data2.index = pd.to_datetime(data2['time'])
    data2['file'] = file
    difference = (max(data.index) - min(data2.index)).days//365 + 1
    data2.index = data2.index + timedelta(days = difference*365)
    data = data.append(data2)



#data = data[data['time'].between('2018-05-01','2018-08-01', inclusive=False)]
data.drop(columns='time',inplace=True)
data.drop(columns='Unnamed: 0',inplace=True)



import statsmodels.api as sm

# Variables
endog = data['activity']
#exog = sm.add_constant(data.loc['tempC'])
exog = data[['tempC', 'uvIndex', 'cloudcover',
       'visibility', 'humidity', 'windspeedKmph']]

model=sm.tsa.statespace.SARIMAX(endog, exog, order=(1, 1, 1),seasonal_order=(1,1,1,4))
model = sm.tsa.statespace.SARIMAX(endog, exog, order=(1,0,0))
results=model.fit()
print(results.summary())

for i in range(0,1200,4):
    data['forecast']=results.predict(start=i,end=i+4,dynamic=True)
print(data['forecast'])
data[['activity','forecast']].plot(figsize=(12,8))
plt.clf()


# find anomalies from model
squared_errors = results.resid ** 2
def find_anomalies(squared_errors):
    threshold = np.mean(squared_errors) + np.std(squared_errors)
    predictions = (squared_errors >= threshold).astype(int)
    return predictions, threshold

predictions, threshold = find_anomalies(squared_errors)
threshold = np.mean(squared_errors) + (1 * np.std(squared_errors))
data['anomaly'] = predictions


for file in files:
    data_file = pd.read_csv(directory+file)
    data_file['arima_anomaly'] = list(data.loc[data['file'] == file]['anomaly'])
    data_file.to_csv(directory+file)



