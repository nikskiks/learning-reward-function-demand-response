import datetime
import pandas as pd
import numpy as np

"""
Loads are grouped in three categories:
    (1) time-shiftable : EV, washing machine, dish washer and dryer. Once turned 
        on, they use constat power.
        EV consumes 4 kW, WM 1 kW, DW 2 kW, DRY 2 kW
    (2) power curtailable : AC - float for demand
    (3) non-shiftable : float for demand.
    
(1) and (2) are categorized as shiftable.
"""

def load_requests():
    """
    The function reads consumption data from a csv file
    
    Returns
    -------
    df : DataFrame (875916x8)
        columns: time, dataid, total, devices (len(DEVICES) types)
        rows: 15 minute periods of year 2018 for each user (NUM_AGENTS)
    """
    df = pd.read_csv(
        'data/15minute_data_austin_fixed_consumption.csv',
        parse_dates=['time'],
        index_col=['time']
    )
    return df


def load_day(df, day, max_steps):
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
    start_date = datetime.datetime.strptime('{} {}'.format(day, 2018), '%j %Y')
    end_date = start_date + time_delta
    df = df.loc[(df.index >= start_date) & (df.index < end_date)]
    return df


def get_device_demands(df, agent_ids, day, timestep):
    """
    The function extracts devices' consumption for each user at a single 15-minute 
    period.
    
    Parameters
    ----------
    df : DataFrame (2400x8)
        a certain day data for each user; from load_day function
    agent_ids : list of ints
                agent IDs
    day : int
        number order of the day - day: 177 -> start_date: 2018-06-26
    timestep : int
            number order of 15 minute period, e.g. if it is 7th 15 minute period 
            timestep = 7 -> minutes = 105 -> time_delta = 1h45mins 
                
    Returns
    -------
    df : DataFrame (NUM_AGENTSx8)
        columns: time, dataid, total, devices (len(DEVICES) types)
        rows: each user
    """
    minutes = timestep * 15
    time_delta = pd.to_timedelta(minutes, 'm')
    start_date = datetime.datetime.strptime('{} {}'.format(day, 2018), '%j %Y')
    time = start_date + time_delta
    df = df.loc[(df['dataid'].isin(agent_ids)) & (df.index == time)]
    return df


def get_peak_demand(df):
    """
    The function extracts total peak consumption in that day for all users together

    Parameters
    ----------
    df : DataFrame (2400x8)
        Data for a certain day for each user (from load_day function)
 
    Returns
    -------
    float           
    """
    df = df.groupby(pd.Grouper(freq='15Min')).sum()
    return df['total'].max()


def load_baselines():
    """
    The function load the baseline load for each customer calculated as the 
    average load of the last 10 day.

    Returns
    -------
    numpy array of floats (NUM_AGENTS x 274 x TIME_STEPS_TRAIN)
        Returns the values of customer baseline load for each customer for each 
        day and timestep. The first element (0) is corresponding to the first day 
        of training (91 day of the year).
    """
    return np.load('data/baselines_regr_temp_correction.npy')