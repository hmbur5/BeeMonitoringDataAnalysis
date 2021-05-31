from WorldWeatherPy import HistoricalLocationWeather
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# weather data from https://www.worldweatheronline.com/developer/api/docs/historical-weather-api.aspx
# use api to download historical weather data from the same place and time
keys = ['f2f090e1b01d4d7ea1435335211404', 'd6c47801209e43a6b2150339211105', 'd901813a0ad147ca829234002210505', 'a4e8b13df3d34c208ea22145210605']
api_key = keys[0]


df = pd.read_pickle('daily_fluctuations.pkl')
print(df.columns)
try:
    weather = pd.read_pickle('weather_warwick.pkl')
except:
    # get forecast weather
    first_date = min(df.index).strftime('%Y-%m-%d')
    end_date = max(df.index).strftime('%Y-%m-%d')
    weather = HistoricalLocationWeather(api_key, 'Warwick+Queensland+Australia', first_date, end_date, 1).retrieve_hist_data()
    weather.to_pickle('weather_warwick.pkl')


weather['cloudcover'] = pd.to_numeric(weather['cloudcover'])
weather['tempC'] = pd.to_numeric(weather['tempC'])
weather['maxtempC'] = pd.to_numeric(weather['maxtempC'])
weather['humidity'] = pd.to_numeric(weather['humidity'])
# interpolate data so time steps are even
upsampled = weather.resample('15min').mean()
interpolated = upsampled.interpolate(method='linear')
#print(interpolated['tempC'])


plt.plot(df.index, df['Hive 1:weights rolling'])
rolling_temp = interpolated['humidity'].rolling(window=96).mean()
plt.plot(interpolated.index, rolling_temp)
plt.legend(['weight','temp'])
plt.show()