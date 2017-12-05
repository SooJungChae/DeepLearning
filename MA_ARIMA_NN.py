from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import scipy 
import numpy as np
import os
import csv
import math
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import accuracy_score
from statsmodels.sandbox.regression.predstd import wls_prediction_std

Train_X = []; Train_Y = []
Test_X = []; Test_Y = []

fName = 'Sales_By_Weather_Time_Series_soo.csv'

df = pd.read_csv(fName)

h = 0
row_count = 1133
Train_cnt = round(row_count * 0.9)
Test_cnt = round(row_count * 0.1)

# Store Train and Test Data From CSV file
for record in reader:
  
  data = record[:-1]
  
  if(h == 1132):
    break
  if (h > Train_cnt):
    Test_X.append(list(map(float, data)))
    Test_Y.append(int(record[-1]))	# SaleQty
    Date_Test.append(record[0])
  else:
    Train_X.append(list(map(float, data)))
    Train_Y.append(int(record[-1]))
    Date_Train.append(record[0])
  
  h = h + 1
