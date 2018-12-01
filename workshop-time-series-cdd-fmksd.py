import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from math import sqrt
import warnings
from scipy.stats import boxcox
from statsmodels.tsa.arima_model import ARIMAResults
from sklearn.metrics import mean_squared_error
from math import sqrt
from math import exp
from math import log


# series = pd.Series.from_csv('robberies.csv', header=0)
series = pd.read_csv('robberies.csv', index_col=0)

split_point = len(series) - 12
dataset, validation = series[0:split_point], series[split_point:(len(series))-1]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))

dataset.to_csv('dataset_robberies.csv')
validation.to_csv('validation_robberies.csv')

asdf = pd.read_csv('dataset_robberies.csv')
asdf.plot()
plt.show()

print(asdf.describe())

rolmean = asdf.rolling(window=20).mean()
rolstd = asdf.rolling(window=20).std()
#Plot rolling statistics:
orig = plt.plot(asdf, color='blue',label='Original')
mean = plt.plot(rolmean, color='red', label='Rolling Mean')
std = plt.plot(rolstd, color='black', label = 'Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show(block=False)