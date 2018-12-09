# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score

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

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Importing the model:
from sklearn.ensemble import RandomForestClassifier


# Initiating the model:
lr = RandomForestClassifier()

scores = cross_val_score(lr, X_train, y_train, scoring='accuracy' ,cv=10).mean()
print("The mean accuracy with 10 fold cross validation is %s" % round(scores*100,2))
