# Setup

import pandas as pd
import numpy as np
from scipy.stats import mode as mode
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('data/hour.csv')

data['days_since_start'] = (pd.to_datetime(data['dteday']) - min(pd.to_datetime(data['dteday']))).dt.days
data = data.drop(['instant', 'dteday', 'casual', 'registered'], axis=1)

#### de-normalize features (for more intuitive plot axes in the talk)
#- temp : Normalized temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-8, t_max=+39 (only in hourly scale)
#- atemp: Normalized feeling temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-16, t_max=+50 (only in hourly scale)
#- hum: Normalized humidity. The values are divided to 100 (max)
#- windspeed: Normalized wind speed. The values are divided to 67 (max)

data['temp'] = data['temp'] * 39 - 8
data['atemp'] = data['atemp'] * 50 - 16
data['hum'] = data['hum'] * 100
data['windspeed'] = data['windspeed'] * 67

#### Pretty factor levels and feature names

data = data.replace({
    'season': {1: 'spring', 2: 'summer', 3: 'autumn', 4: 'winter'},
    'holiday': {0: 'no', 1: 'yes'},
    'workingday': {0: 'no', 1: 'yes'},
    'weekday': {0: 'sunday', 1: 'monday', 2: 'tuesday', 3: 'wednesday', 4: 'thursday', 5: 'friday', 6: 'saturday'},
    'weathersit': {1: 'clear', 2: 'mist', 3: 'light rain', 4: 'heavy rain'},
    'mnth': {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
              7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'},
    'yr': {0: 2011, 1: 2012}
}).rename({
    'atemp': 'feel_temp',
    'weathersit': 'weather',
    'mnth': 'month',
    'yr': 'year',
    'hum': 'humidity',
    'cnt': 'count'
}, axis=1)

data = data.drop('month', axis=1)

# Aggregate data to per-day

data = data.groupby('days_since_start').agg({
    'season': lambda x: mode(x)[0],
    'year': 'mean',
    'holiday': lambda x: mode(x)[0],
    'weekday': lambda x: mode(x)[0],
    'workingday': lambda x: mode(x)[0],
    'weather': lambda x: mode(x)[0],
    'temp': 'mean',
    'feel_temp': 'mean',
    'humidity': 'mean',
    'windspeed': 'mean',
    'count': 'sum'
})

# And for prettiness in this talk, I'll round floats to 1 digit:
data = data.round(1)

#### transform factors to dummies

def dummify_dataset(df, column):       
    # I don't drop the first category. Collinearity is "ok" in tree-based models, and I want to
    # keep them for interpretability
    df = pd.concat([df, pd.get_dummies(df[column], prefix=column, drop_first=False)],axis=1)
    df = df.drop([column], axis=1)
    return df

dummy_data = data.copy()

for col in data.columns[data.dtypes == 'object']:
    dummy_data = dummify_dataset(dummy_data, col)
