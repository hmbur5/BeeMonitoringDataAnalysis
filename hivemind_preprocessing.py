import pandas as pd

reading = pd.read_csv('hivemind_data/reading.csv')
event = pd.read_csv('hivemind_data/event.csv')
event['event_id'] = event['id']
event = event[['event_id', 'location', 'weather_data']]
event_records = pd.merge(reading,event, on='event_id', how='outer', validate = 'many_to_one')
# only using events that have location data
event_records = event_records.loc[event_records['location'].notnull()]
event_records.to_csv('hivemind_data/event_records1.csv')
exit()

# find reading and matching event, then put into a single, complete dataframe
event_records = None

prev_index = 0
for i1 in range(reading.shape[0]):
    # skip any hidden readings (which should be considered deleted), or readings of battery
    if reading['hidden'][i1]=='t':
        continue
    if reading['type'][i1]=='battery':
        continue

    for i2 in range(prev_index,event.shape[0]):
        if reading['hub_id'][i1] != event['device_id'][prev_index] and reading['hub_id'][i1] == event['device_id'][i2]:
            prev_index=i2
            print(prev_index)
        if reading['hub_id'][i1] > event['device_id'][i2]:
            break

        if reading['event_id'][i1] == event['id'][i2]:
            print('match')
            event_match = event.iloc[[i2]]
            reading_match = reading.iloc[[i1]]
            #print(event_match.columns)
            #print(reading_match.columns)
            #print(reading_match)
            for column in event_match.columns:
                if column not in reading_match.columns:
                    reading_match[column] = [event_match[column][i2]]
            #print(reading_match['type'])
            #print(reading_match['location'])
            if event_records is None:
                event_records = reading_match
            else:
                event_records = event_records.append(reading_match)
            break

    try:
        if event_records.shape[0]%1000==0:
            event_records.to_csv('hivemind_data/event_records.csv')
    except AttributeError:
        pass