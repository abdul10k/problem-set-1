'''
PART 1: ETL the two datasets and save each in `data/` as .csv's
'''

import pandas as pd
import os 


# Create 'data' directory if it doesn't exist
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(data_dir, exist_ok=True)




pred_universe_raw = pd.read_csv('https://www.dropbox.com/scl/fi/69syqjo6pfrt9123rubio/universe_lab6.feather?dl=1')
arrest_events_raw = pd.read_csv('https://www.dropbox.com/scl/fi/wv9kthwbj4ahzli3edrd7/arrest_events_lab6.feather?dl=1')
pred_universe_raw['arrest_date_univ'] = pd.to_datetime(pred_universe_raw.filing_date)
arrest_events_raw['arrest_date_event'] = pd.to_datetime(arrest_events_raw.filing_date)
pred_universe_raw.drop(columns=['filing_date'], inplace=True)
arrest_events_raw.drop(columns=['filing_date'], inplace=True)
pred_universe_raw.to_csv('data/pred_universe_raw.csv', index=False)
arrest_events_raw.to_csv('data/arrest_events_raw.csv', index=False)