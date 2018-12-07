# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

# Importing the dataset
dataset = pd.read_csv('breast-cancer-wisconsin.data', index_col=0)
dataset.replace('?', '1', inplace=True) #-9999 biar outlier, gak masuk ke grafik 
X = dataset.iloc[:, 1:10].values # 11 karena ada class
y = dataset.iloc[:, 9].values # 10 karena tidak ada kelas 

print("\n \t The data frame has {0[0]} rows and {0[1]} columns. \n".format(dataset.shape))
dataset.info()

print(dataset.head(3))


#visualizing data 
features_mean = list(dataset.columns[0:10])
print(features_mean)
plt.figure(figsize=(10,10))
sns.heatmap(dataset[features_mean].corr(), annot=True, square=True, cmap='coolwarm')
plt.show()


print(dataset.columns)

sns.pairplot(dataset, hue='class', vars = ['clump_thickness', 'unif_cell_size','unif_cell_shape', 'marg_adhesion'])
plt.show()

#Print benign and malign cancer 

sns.countplot(dataset['class'], label = "Hitung")
plt.show()

# # Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

X = dataset.drop(['class'], axis = 1) # We drop our "target" feature and use all the remaining features in our dataframe to train the model.
print(X.head())
y = dataset['class']
print('\n')
print(y.head())


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


# check the test and train data 
print ('The size of our training "X" (input features) is', X_train.shape)
print ('\n')
print ('The size of our testing "X" (input features) is', X_test.shape)
print ('\n')
print ('The size of our training "y" (output feature) is', y_train.shape)
print ('\n')
print ('The size of our testing "y" (output features) is', y_test.shape)

from sklearn.svm import SVC
svc_model = SVC()

print(svc_model.fit(X_train, y_train))
y_predict = svc_model.predict(X_test)


# visuaslisai preidksi menggunakan svm dengan confusion matrix 
from sklearn.metrics import classification_report, confusion_matrix
cm = np.array(confusion_matrix(y_test, y_predict, labels=[2,4]))
confusion = pd.DataFrame(cm, index=['is_cancer', 'is_healthy'], columns=['predicted_cancer','predicted_healthy'])
print(confusion)
sns.heatmap(confusion, annot=True)
plt.show()

print(classification_report(y_test, y_predict))

# visualisasi dengan grid 
# creating a meshgrid
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h=0.05
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
xy_mesh=np.c_[xx.ravel(), yy.ravel()]
Z = svc_model.predict(xy_mesh)
Z = Z.reshape(xx.shape)

#plotting data on decision boundary
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xlabel('PC1');plt.ylabel('PC2')
plt.title('SVC')
plt.show()