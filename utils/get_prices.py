# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 17:06:17 2023

@author: Korisnik
"""

import datetime
from datetime import datetime
import pandas as pd
import numpy as np

def load_prices(fif=False):
    df = pd.read_csv(
        'Enter path for prices files',
    )
    df = df[df['Zone'] == "LZ_HOUSTON"].values
    df = fif_min_gran(df)
    
    return df

def fif_min_gran(df):
    new_dates = []
    new_vals = []
    for i in range(0, len(df)):
        for j in range(0, 4):
            if j==0:
                date = df[i][0][:-2] + '00'
                new_dates.append(datetime.strptime(date, '%m/%d/%Y %H:%M'))
            else:
                date = df[i][0][:-2] + str(15*j)
                new_dates.append(datetime.strptime(date, '%m/%d/%Y %H:%M'))
            new_vals.append(df[i][1])
    new_df = np.hstack((np.array(new_dates).reshape((-1,1)), np.array(new_vals).reshape((-1,1))))
    new_df = pd.DataFrame(new_df, columns = ['Date','Price'])
    return new_df

def load_prices_day(df, day, max_steps):
    """
    The function extracts a certain day from the whole yearly dataset for each 
    user.
    
    Returns
    -------
    df : DataFrame (2400x8)
        columns: time, dataid, total, devices (len(DEVICES) types)
        rows: 15 minute periods of a day for each user (TIME_STEPS_TRAIN time 
            steps (1 day) * NUM_AGENTS users, 96 x 25 = 2400)
    """
    minutes = max_steps * 15
    time_delta = pd.to_timedelta(minutes, 'm')
    start_date = datetime.strptime('{} {}'.format(day, 2018), '%j %Y')
    end_date = start_date + time_delta
    df = df.loc[(df['Date'] >= start_date) & (df['Date'] < end_date)]
    df = df.set_index('Date')
    return df

df = load_prices(fif=True)
dan = load_prices_day(df, 91, 96)
