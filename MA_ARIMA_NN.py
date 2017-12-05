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


''''================================
    Step1 - Make data stationary
    I followed codes from reference [1]
================================''''

ts_log = np.log(ts)
df_log = pd.DataFrame(ts_log)
noinf_df_log = df_log.replace([np.inf, -np.inf], np.nan).dropna()
ts_log = noinf_df_log['SaleQty']

# Estimating & Eliminating trend. Moving Average
moving_avg = ts_log.rolling(window=2,center=False).mean()

plt.figure(figsize=(10,6))
plt.plot(ts_log)
plt.plot(moving_avg, color='red')
plt.show()

# Check data is stationary
ts_log_ma_diff = ts_log - moving_avg
ts_log_ma_diff.head(12)


def test_stationarity(timeseries):
	from statsmodels.tsa.stattools import adfuller
	
	#Determing rolling statistics
	rolmean = pd.rolling_mean(timeseries, window=12)
	rolstd = pd.rolling_std(timeseries, window=12)
	
	#Plot rolling statistics:
	plt.figure(figsize=(10,6))
	orig = plt.plot(timeseries, color='blue',label='Original')
	mean = plt.plot(rolmean, color='red', label='Rolling Mean')
	std = plt.plot(rolstd, color='black', label = 'Rolling Std')
	plt.legend(loc='best')
	plt.title('Rolling Mean & Standard Deviation')
	plt.show(block=False)
	
	#Perform Dickey-Fuller test:
	print ('Results of Dickey-Fuller Test:')
	dftest = adfuller(timeseries, autolag='AIC')
	print(dftest)
	dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
	for key,value in dftest[4].items():
		dfoutput['Critical Value (%s)'%key] = value
	print (dfoutput)


ts_log_ma_diff.dropna(inplace=True)
test_stationarity(ts_log_ma_diff)
# Compare Test Statistic, Critical Values

# ACF and PACF plots
from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(ts_log_ma_diff, nlags=20)
lag_pacf = pacf(ts_log_ma_diff, nlags=20, method='ols')

#Plot ACF: 
plt.figure(figsize=(10,6))
plt.subplot(121) 
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_ma_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_ma_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_ma_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_ma_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()

plt.show()

''''================================
    Step2 - Make ARIMA model
    I followed codes from reference [1]
================================''''

from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(ts_log_ma_diff, order=(1,0,1)) # order came out from ACF, PACF plot data
results_ARIMA = model.fit(disp=-1)

plt.figure(figsize=(10,6))
plt.plot(ts_log_ma_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues.values - ts_log_ma_diff.values) ** 2))
plt.show()



''''
Working on...
''''




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
