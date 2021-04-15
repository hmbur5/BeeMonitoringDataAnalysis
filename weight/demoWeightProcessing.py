import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

#2000-01-01 11:00:53
def parser(x):
    return datetime.strptime(x,'%Y-%m-%d %H:%M:%S')


df = pd.read_csv('Hivemind data - Demo - 2021-04-15 14.22.csv', parse_dates=[0], date_parser=parser)
print(df)
df = df.set_index('date/time (Australia/Melbourne)')

# there is a duplicate row in the demo data, this shouldn't happen in reality
print(df[df.index.duplicated()])
df = df.loc[~df.index.duplicated(),:]
plt.plot(df.index, df['Hive 1:weights'])

# interpolate data so time steps are even
upsampled = df.resample('15min').mean()
interpolated = upsampled.interpolate(method='linear')
print(interpolated['Hive 1:weights'])

# moving average filter
interpolated['Hive 1:weights rolling'] = interpolated['Hive 1:weights'].rolling(window=96).mean()

plt.plot(interpolated.index, interpolated['Hive 1:weights rolling'])



# low pass filter

from scipy.signal import butter, lfilter, freqz

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

interpolated['Hive 1:weights LPF'] = butter_lowpass_filter(interpolated['Hive 1:weights'], 48, 10000)

plt.plot(interpolated.index, interpolated['Hive 1:weights LPF'])
plt.legend(['Raw', 'Moving average','LPF'])
plt.ylabel('Weight (kg)')
plt.show()
