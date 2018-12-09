from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
digits = load_digits()
kmeans = KMeans(n_clusters=10)
clusters = kmeans.fit_predict(digits.data)
print(kmeans.cluster_centers_.shape)

#------------------------------------------------------------
# visualize the cluster centers
fig = plt.figure(figsize=(8, 3))
for i in range(10): 
    ax = fig.add_subplot(2, 5, 1 + i)
    ax.imshow(kmeans.cluster_centers_[i].reshape((8, 8)),
              cmap=plt.cm.binary)
from sklearn.manifold import Isomap
X_iso = Isomap(n_neighbors=10).fit_transform(digits.data)

#------------------------------------------------------------
# visualize the projected data
fig, ax = plt.subplots(1, 2, figsize=(8, 4))

ax[0].scatter(X_iso[:, 0], X_iso[:, 1], c=clusters)
ax[1].scatter(X_iso[:, 0], X_iso[:, 1], c=digits.target)
plt.show()

#------------------------------------------------------------
# titanic data 

import os
import pandas as pd

titanic = pd.read_csv('titanic3.csv')
print(titanic.columns)

labels = titanic.survived.values
features = titanic[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]

print(features.head())
print(pd.get_dummies(features).head())

features_dummies = pd.get_dummies(features, columns=['pclass', 'sex', 'embarked'])
print(features_dummies.head(n=16))
data = features_dummies.values

import numpy as np
np.isnan(data).any()
print(np.isnan(data).any())
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer


train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state=0)

imp = Imputer()
imp.fit(train_data)
train_data_finite = imp.transform(train_data)
test_data_finite = imp.transform(test_data)
print(np.isnan(train_data_finite).any())


from sklearn.dummy import DummyClassifier

clf = DummyClassifier('most_frequent')
clf.fit(train_data_finite, train_labels)
print("Prediction accuracy: %f"
      % clf.score(test_data_finite, test_labels))

train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, random_state=0)

imp = Imputer()
imp.fit(train_data)
train_data_finite = imp.transform(train_data)
test_data_finite = imp.transform(test_data)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression().fit(train_data_finite, train_labels)
print("logistic regression score: %f" % lr.score(test_data_finite, test_labels))

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=500, random_state=0).fit(train_data_finite, train_labels)
print("random forest score: %f" % rf.score(test_data_finite, test_labels))

features_dummies_sub = pd.get_dummies(features[['pclass', 'sex', 'age', 'sibsp', 'fare']])
data_sub = features_dummies_sub.values

train_data_sub, test_data_sub, train_labels, test_labels = train_test_split(data_sub, labels, random_state=0)

imp = Imputer()
imp.fit(train_data_sub)
train_data_finite_sub = imp.transform(train_data_sub)
test_data_finite_sub = imp.transform(test_data_sub)
                                         
lr = LogisticRegression().fit(train_data_finite_sub, train_labels)
print("logistic regression score w/o embark, parch: %f" % lr.score(test_data_finite_sub, test_labels))
rf = RandomForestClassifier(n_estimators=500, random_state=0).fit(train_data_finite_sub, train_labels)
print("random forest score w/o embark, parch: %f" % rf.score(test_data_finite_sub, test_labels))