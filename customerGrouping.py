import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

# each hive from hivemind data has a different site id?

event_record = pd.read_csv('hivemind_data/event_records1.csv')
event_record = event_record.sort_values(by=['site_id', 'device_id'])
event_record.index = range(len(event_record.index))
event_record['time'] = pd.to_datetime(event_record['time'], utc=True)

variables = ['weights', 'outside_temp', 'humidity', 'temp']
'''
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


#with open('siteDevices.pickle', 'wb') as handle:
#    pickle.dump(siteDevices, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
        ot = siteDevices[site_number]['outside_temp']
        intersection = list(set(temp) & set(ot))
        if len(intersection)>0:
            site_list.append(site_number)
            #for variable in variables:
            #    climateRecords[variable][site_number] = siteRecords[variable][site_number]
    except:
        pass

print(site_list)

'''


siteRecords = pd.DataFrame()
site_list = [34.0, 429.0, 1439.0, 1677.0, 4483.0, 4487.0, 4488.0, 4491.0, 4493.0, 5684.0, 6786.0, 1443.0, 5623.0, 5924.0]
event_record = event_record[event_record['site_id'].notna()]
siteRecords = event_record[event_record['site_id'].isin(site_list)]


variables = ['temp', 'outside_temp']
for site_number in site_list:
    site_df = siteRecords.loc[(siteRecords['site_id'] ==site_number)]
    devices = site_df['device_id'].unique()
    max = 0
    max_device = None
    max_dataframe = None

    for device in devices:
        for variable in variables:
            df = site_df.loc[(site_df['device_id'] == device) & (site_df['type'] ==variable)]
            df = df[['time','value']]
            df = df.set_index('time')
            # interpolate data so time steps are even
            upsampled = df.resample('6H').mean()
            interpolated = upsampled.interpolate(method='linear', limit=2)
            interpolated = interpolated.rename(columns={"value": variable})

            if variable == variables[0]:
                all_measures = interpolated
            else:
                all_measures = pd.merge(all_measures, interpolated, on = 'time', how = 'outer', validate = 'one_to_one')

        if all_measures.dropna().shape[0]>max:
            max = all_measures.dropna().shape[0]
            max_device = device
            max_dataframe = all_measures.dropna()

    plt.scatter(max_dataframe.index, max_dataframe.temp)
    plt.scatter(max_dataframe.index, max_dataframe.outside_temp)
    #plt.show()

    max_dataframe.to_csv(str('temp_data/site'+str(site_number)+'_device'+str(max_device)+'.csv'))


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