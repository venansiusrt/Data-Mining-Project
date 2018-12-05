# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('breast-cancer-wisconsin.data', index_col=0)
dataset.replace('?', '1', inplace=True) #-9999 biar outlier, gak masuk ke grafik 
X = dataset.iloc[:, 1:10].values # 11 karena ada class
y = dataset.iloc[:, 9].values # 10 karena tidak ada kelas 

print("\n \t The data frame has {0[0]} rows and {0[1]} columns. \n".format(dataset.shape))
dataset.info()

print(dataset.head(3))

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

#visualizing data 
features_mean = list(dataset.columns[0:10])
print(features_mean)
plt.figure(figsize=(10,10))
sns.heatmap(dataset[features_mean].corr(), annot=True, square=True, cmap='coolwarm')
plt.show()


print(dataset.columns)

sns.pairplot(dataset, hue='class' ,vars = ['clump_thickness', 'unif_cell_size','unif_cell_shape', 'marg_adhesion'])
plt.show()

print(dataset['class'].value_counts() +  '\n\n\n')



# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)