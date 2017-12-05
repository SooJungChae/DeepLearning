''''
Step1
Make data stationary

Step2
ARIMA

Step3
ANN

[Reference]
[1] A comprehensive beginner’s guide to create a Time Series Forecast (with Codes in Python)
    Link --> https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/
''''

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from datetime import datetime
from matplotlib import pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std

import tensorflow as tf
import scipy 
import numpy as np
import csv
import math
import pandas as pd
import statsmodels.api as sm

fName = 'Sales_By_Weather_Time_Series_soo.csv'
df = pd.read_csv(fName)

df['SaleDate'] = pd.to_datetime(df['SaleDate'], format='%Y%m%d')
new_df = df.set_index('SaleDate')

ts = new_df['SaleQty']


### Step1 - Make data stationary
### I followed codes from reference [1]

ts_log = np.log(ts)
df_log = pd.DataFrame(ts_log)
noinf_df_log = df_log.replace([np.inf, -np.inf], np.nan).dropna()
ts_log = noinf_df_log['SaleQty']

# Estimating & Eliminating trend. Moving Average
moving_avg = ts_log.rolling(window=2,center=False).mean()




Train_X = []; Train_Y = []
Test_X = []; Test_Y = []

row_count = 1133
Train_cnt = round(row_count * 0.9)
Test_cnt = round(row_count * 0.1)

# Store Train and Test Data From CSV file
h = 0
for record in df:
  
  data = record[:-1]
  
  if(h == 1132):
    break
  if (h > Train_cnt):
    Test_X.append(list(map(float, data)))
    Test_Y.append(int(record[-1]))	# SaleQty
    Date_Test.append(record[0]) # SaleDate
  else:
    Train_X.append(list(map(float, data)))
    Train_Y.append(int(record[-1]))
    Date_Train.append(record[0])
  
  h = h + 1
