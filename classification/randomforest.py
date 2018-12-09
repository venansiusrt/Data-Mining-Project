# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('breast-cancer-wisconsin.data')
dataset.replace('?', -99999, inplace=True) #-9999 biar outlier, gak masuk ke grafik 
X = dataset.iloc[:, 1:11].values # 11 karena ada class
y = dataset.iloc[:, 10].values # 10 karena tidak ada kelas 

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print(X_test)