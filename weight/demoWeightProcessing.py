import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import math
from scipy.signal import butter, lfilter
from WorldWeatherPy import HistoricalLocationWeather

# low pass filter
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# load data
df = pd.read_csv('Hivemind data - Demo - 2021-04-15 14.22.csv', parse_dates=[0],
                 date_parser=lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
df = df.set_index('date/time (Australia/Melbourne)')
# deleting first few elements from data as they have a much older time stamp (making it difficult to see graph properly)
df = df[12:]
# also, there is a duplicate row in the demo data, this shouldn't happen in reality - so we delete it.
df = df.loc[~df.index.duplicated(),:]

# plotting raw data
plt.plot(df.index, df['Hive 1:weights'])
plt.ylabel('Weight (kg)')
plt.show()

# dealing with significant jumps in weight from supers
weight = []
relative = -df['Hive 1:weights'][0]
for index,element in enumerate(df['Hive 1:weights']):
    # skip NaN elements
    if math.isnan(element):
        continue
    # if weight change in this time step is greater than 5kg, adjust for this change by using a 'relative' value
    if len(weight)>0:
        if abs(weight[-1]-(element+relative))>3:
            relative += weight[-1]-(element+relative)
    weight.append(element+relative)
    df['Hive 1:weights'][index] = element+relative


# interpolate data so time steps are even
upsampled = df.resample('15min').mean()
interpolated = upsampled.interpolate(method='linear')

# moving average filter on interpolated data
interpolated['Hive 1:weights rolling'] = interpolated['Hive 1:weights'].rolling(window=96).mean()

# low pass filter
# sampling frequency is 96 (samples per day - 1 every 15 minutes). cuttoff frequency is 0.5 (1 per 2 days/48 hours)
interpolated['Hive 1:weights LPF'] = butter_lowpass_filter(interpolated['Hive 1:weights'], cutoff=0.5, fs=96)

# short term weight change by subtracting filtered from unfiltered
interpolated['Hive 1:weights fluctuations'] = interpolated['Hive 1:weights'] - interpolated['Hive 1:weights rolling']


# get corresponding weather data for location and time period
first_date = min(df.index - timedelta(1)).strftime('%Y-%m-%d')
end_date = max(df.index + timedelta(1)).strftime('%Y-%m-%d')
api_key = 'f2f090e1b01d4d7ea1435335211404'
weather = HistoricalLocationWeather(api_key, 'Warwick+Queensland+Australia', first_date, end_date, 1).retrieve_hist_data()
weather = pd.to_numeric(weather['tempC'])
weather = weather.interpolate(method='linear')
# filter fluctuations by getting daily average
temp_davg = weather.resample('D').mean()

# plotting Figure 4
fig, ax1 = plt.subplots()
# plot different weight filters
ax1.plot(interpolated.index, interpolated['Hive 1:weights'])
ax1.plot(interpolated.index, interpolated['Hive 1:weights rolling'])
ax1.plot(interpolated.index, interpolated['Hive 1:weights LPF'])
ax1.legend(['Unfiltered', 'Moving average','LPF'], title='Weight filters', loc=9)
ax1.set_ylabel('Weight (kg)')
# add second axis for temperature
ax2 = ax1.twinx()
ax2.set_ylabel('Daily average temperature (degrees celcius)')
ax2.plot(temp_davg, color='black')
ax1.set_title('Filtering of weight data from beehive in Warwick Queensland')
plt.show()

# plotting Figure 5
plt.plot(interpolated.index, interpolated['Hive 1:weights fluctuations'])
plt.title('Daily weight fluctuations from beehive in Warwick Queensland')
plt.ylabel('Weight (kg)')
plt.show()