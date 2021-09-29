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


data = pd.read_csv(directory+file)
data.index = pd.to_datetime(data['time'])
data['hour'] = data.index.hour
data['file'] = file


print(data.columns)

for file in files[1:]:
    data2 = pd.read_csv(directory+file)
    if 'tempC' not in data2.columns:
        continue
    # add some years to each file (for sufficient gap between hives)
    data2.index = pd.to_datetime(data2['time'])
    data2['file'] = file
    data2['hour'] = data2.index.hour
    difference = (max(data.index) - min(data2.index)).days//365 + 1
    data2.index = data2.index + timedelta(days = difference*365)
    data = data.append(data2)



data.drop(columns='time',inplace=True)
data.drop(columns='Unnamed: 0',inplace=True)

from statsmodels.tsa.seasonal import seasonal_decompose
decompose_data = seasonal_decompose(data['activity'], model="additive", period=4)
decompose_data.plot()
plt.clf()


import statsmodels.api as sm

# checking data is stationary
from statsmodels.tsa.stattools import adfuller
dftest = adfuller(data.activity, autolag = 'AIC')
print("1. ADF : ",dftest[0])
print("2. P-Value : ", dftest[1])
print("3. Num Of Lags : ", dftest[2])
print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
print("5. Critical Values :")
for key, val in dftest[4].items():
    print("\t",key, ": ", val)

# endogenous and exogenous variables
endog = data['activity']
#exog = sm.add_constant(data.loc['tempC'])
exog = data[['tempC', 'uvIndex', 'cloudcover',
       'visibility', 'humidity', 'windspeedKmph']]


# find optimal q, p and d using grid search
import itertools

# Define the p, d and q parameters to take any value between 0 and 3 (exclusive)
p = d = q = range(0, 3)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))
pqds = [(x[0], x[1], x[2], 4) for x in list(itertools.product(p, d, q))]


### Run Grid Search ###

# Note: this code will take a while to run

# Define function
def sarimax_gridsearch(endog, exog, pdq, pdqs, maxiter=50):
    # Run a grid search with pdq parameters and get the best BIC value
    ans = []
    for comb in pdq:
        for combs in pdqs:
            print('hi')
            mod = sm.tsa.statespace.SARIMAX(endog, exog, order=comb, seasonal_order=combs)

            output = mod.fit(maxiter=maxiter)
            ans.append([comb, combs, output.bic])
            print('SARIMAX {} x {}12 : BIC Calculated ={}'.format(comb, combs, output.bic))

    # Find the parameters with minimal BIC value

    # Convert into dataframe
    ans_df = pd.DataFrame(ans, columns=['pdq', 'bic'])

    # Sort and return top 5 combinations
    ans_df = ans_df.sort_values(by=['bic'], ascending=True)[0:5]

    return ans_df


# grid search
print(sarimax_gridsearch(endog, exog, pdq, pqds))
exit()

# order = (0, 1, 2) gives the largest bic
model = sm.tsa.statespace.SARIMAX(endog, exog, order=(0, 1, 2), seasonal_order=(1,1,1,4))
results=model.fit()
print(results.summary())

for i in range(0,1200,4):
    data['forecast']=results.predict(start=i,end=i+4,dynamic=True)
data[['activity','forecast']].plot(figsize=(12,8))
plt.clf()


# find anomalies from model
# using error relative to the change in activity at this time step
# + 1 to deal with /0 error.
errors = (results.resid / (np.abs(data['activity'].diff(periods=1))+1))**2

data['anomaly'] = errors

#threshold = np.mean(errors) + (0.5 * np.std(errors))

threshold = np.percentile(errors.values[~np.isnan(errors.values)], 98)
print(threshold)
predictions = (errors >= threshold).astype(int)
data['anomaly'] = predictions
print(np.sum(predictions)/len(predictions))


for file in files:
    print(file)
    data_file = pd.read_csv(directory+file)
    if 'tempC' not in data_file.columns:
        continue
    data_file = data_file.loc[:, ~data_file.columns.str.contains('^Unnamed')]
    data_file['arima_anomaly'] = list(data.loc[data['file'] == file]['anomaly'])
    data_file.to_csv(directory+file)



