import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from datetime import timedelta
from socket import *
import time

# each hive from hivemind data has a different site id?

event_record = pd.read_csv('hivemind_data/event_records1.csv')
event_record = event_record.sort_values(by=['site_id', 'device_id'])
event_record.index = range(len(event_record.index))
event_record['time'] = pd.to_datetime(event_record['time'], utc=True)

'''
variables = ['activity','weights', 'outside_temp', 'humidity', 'temp']

current_site = 0
siteRecords = {}
siteDevices = {}
for variable in variables:
    siteRecords[variable] = {}
for variable in variables:
    for i1 in range(event_record.shape[0]):
        # just looking at the data for a given variable
        if event_record['type'][i1]!= variable:
            continue
        if event_record['site_id'][i1]!=current_site:
            current_site = event_record['site_id'][i1]
        try:
            siteRecords[variable][current_site] = siteRecords[variable][current_site].append(event_record.iloc[i1])
        except KeyError:
            siteRecords[variable][current_site] = event_record.iloc[i1]
        try:
            if event_record['device_id'][i1] not in siteDevices[current_site][variable]:
                siteDevices[current_site][variable] = siteDevices[current_site][variable] + [event_record['device_id'][i1]]
        except KeyError:
            try:
                siteDevices[current_site][variable]=[event_record['device_id'][i1]]
            except KeyError:
                siteDevices[current_site] = {}
                siteDevices[current_site][variable] = [event_record['device_id'][i1]]
        if i1%1000==0:
            print(i1)


with open('siteDevices.pickle', 'wb') as handle:
    pickle.dump(siteDevices, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open('siteDevices.pickle', 'rb') as handle:
    siteDevices = pickle.load(handle)

print(siteDevices[34.0])

climateRecords = {}
for variable in variables:
    climateRecords[variable] = {}

site_list = []
for site_number in siteDevices.keys():
    try:
        hum = siteDevices[site_number]['humidity']
        temp = siteDevices[site_number]['temp']
        act = siteDevices[site_number]['activity']
        intersection = list(set(temp) & set(act))
        if len(intersection)>0:
            site_list.append(site_number)
            #for variable in variables:
            #    climateRecords[variable][site_number] = siteRecords[variable][site_number]
    except:
        pass

print(site_list)
'''



siteRecords = pd.DataFrame()
site_list = [34.0, 68.0, 429.0, 682.0, 1021.0, 1046.0, 1047.0, 1048.0, 1439.0, 1443.0, 1677.0, 2604.0, 2754.0, 4483.0, 4484.0, 4487.0, 4488.0, 4491.0, 4493.0, 4494.0, 4495.0, 4496.0, 4869.0, 4873.0, 4888.0, 4890.0, 4906.0, 4907.0, 4908.0, 5018.0, 5092.0, 5093.0, 5095.0, 5147.0, 5167.0, 5168.0, 5170.0, 5179.0, 5348.0, 5362.0, 5432.0, 5433.0, 5435.0, 5491.0, 5561.0, 5563.0, 5606.0, 5607.0, 5608.0, 5617.0, 5619.0, 5620.0, 5623.0, 5624.0, 5625.0, 5628.0, 5637.0, 5659.0, 5660.0, 5662.0, 5664.0, 5665.0, 5666.0, 5667.0, 5668.0, 5669.0, 5670.0, 5675.0, 5676.0, 5677.0, 5684.0, 5688.0, 5689.0, 5895.0, 5898.0, 5924.0, 5925.0, 5926.0, 6124.0, 6125.0, 6463.0, 6645.0, 6786.0, 6787.0, 6788.0, 6790.0]
event_record = event_record[event_record['site_id'].notna()]
siteRecords = event_record[event_record['site_id'].isin(site_list)]

from WorldWeatherPy import HistoricalLocationWeather
keys = ['1dcd3b829f5c4fa9b6514402211308']
api_key = keys[0]

variables = ['activity', 'temp']
for site_number in site_list:
    site_df = siteRecords.loc[(siteRecords['site_id'] ==site_number)]
    devices = site_df['device_id'].unique()
    #max = 0
    #max_device = None
    #max_dataframe = None

    for device in devices:
        for variable in variables:
            df = site_df.loc[(site_df['device_id'] == device) & (site_df['type'] ==variable)]
            if 'location' in df.columns and len(df['location'].values)>0:
                null = None
                location = eval(df['location'].values[0])
                coordinates = str(round(location[0],3)) +','+ str(round(location[1],3))
                print(coordinates)
            else:
                coordinates=None
            df = df[['time','value']]
            df = df.set_index('time')
            # interpolate data so time steps are even
            upsampled = df.resample('6H').mean()
            interpolated = upsampled.interpolate(method='linear', limit=2)
            interpolated = interpolated.rename(columns={"value": variable})

            if variable == variables[0]:
                all_measures = interpolated

                # get current weather conditions at location and time - need to deal with changing locations
                if coordinates is not None and coordinates!='0.0,0.0':
                    first_date = min(df.index).strftime('%Y-%m-%d')
                    end_date = (max(df.index) + timedelta(days=1)).strftime('%Y-%m-%d')
                    print(coordinates)
                    weather=None
                    while weather is None:
                        try:
                            weather = HistoricalLocationWeather(api_key, coordinates, first_date, end_date,
                                                        1).retrieve_hist_data()
                        except timeout:
                            time.sleep(10)
                    weather['time'] = pd.to_datetime(weather.index, utc=True)
                    weather = weather[['time', 'tempC', 'uvIndex', 'cloudcover', 'visibility', 'humidity', 'windspeedKmph']]
                    all_measures = pd.merge(all_measures, weather, on='time', how='inner', validate='one_to_many')

            else:
                all_measures = pd.merge(all_measures, interpolated, on = 'time', how = 'outer', validate = 'one_to_one')

        if all_measures.dropna().shape[0]>5:
            all_measures = all_measures.dropna()
            all_measures.to_csv(str('activity_temp_data/site' + str(site_number) + '_device' + str(device) + '.csv'))

        '''if all_measures.dropna().shape[0]>max:
            max = all_measures.dropna().shape[0]
            max_device = device
            max_dataframe = all_measures.dropna()

    max_dataframe.to_csv(str('activity_data/site'+str(site_number)+'_device'+str(max_device)+'.csv'))'''

'''
good_sites = []
for site_number in siteDevices.keys():
    try:
        site_number = site_number/1
        #print(siteDevices[site_number]['weights'])
        if len(siteDevices[site_number]['weights'])==1:
            if len(siteDevices[site_number]['humidity'])==1:
                good_sites.append(site_number)
    except:
        pass
#exit()
print('hello')
print(len(good_sites))


goodDataDictionary = {}
for variable in variables:
    goodDataDictionary[variable] = {}

for variable in variables:
    for site_number in siteRecords[variable].keys():
        if site_number in good_sites:
            goodDataDictionary[variable][site_number] = siteRecords[variable][site_number]

with open('goodSiteRecords.pickle', 'wb') as handle:
    pickle.dump(goodDataDictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)



with open('goodSiteRecords.pickle', 'rb') as handle:
    goodDataDictionary = pickle.load(handle)

for variable in goodDataDictionary.keys():
    print(goodDataDictionary[variable].keys())
    for key in goodDataDictionary[variable].keys():
        pass
        #print(goodDataDictionary[variable][key])
plt.scatter(goodDataDictionary['weights'][key].time, goodDataDictionary['weights'][key].value)
plt.show()'''