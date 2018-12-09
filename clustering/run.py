import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import matplotlib.cm as cm

from sklearn import preprocessing
# from subprocess import check_outputs



# dataset
data = pd.read_csv('../classification/breast-cancer-wisconsin.data', index_col = 0)
data.replace('?', -99999, inplace=True) 
print(data.head())
print(data.shape)

# scaling 
datas = pd.DataFrame(preprocessing.scale(data.iloc[:,1:11]))
datas.columns = list(data.iloc[:,1:11].columns)
print(datas.head())
print(datas.shape)

# remove the class, biar classnya jadi target 
data_drop = datas.drop('class',axis=1)
X = data_drop.values
print(data_drop.head())

#Creating a 2D visualization to visualize the clusters
from sklearn.manifold import TSNE
tsne = TSNE(verbose=1, perplexity=40, n_iter= 4000)
Y = tsne.fit_transform(X)
print("Y for 2d visualization the cluster", "\n", Y)

#K Means
#Cluster using k-means
from sklearn.cluster import KMeans
kmns = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto')
kY = kmns.fit_predict(X)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

ax1.scatter(Y[:,0],Y[:,1],  c=kY, cmap = "jet", edgecolor = "None", alpha=0.35)
ax1.set_title('k-means clustering plot')

ax2.scatter(Y[:,0],Y[:,1],  c = datas['class'], cmap = "jet", edgecolor = "None", alpha=0.35)
ax2.set_title('Actual clusters')
plt.show()

#spectral clustering 
from sklearn.cluster import SpectralClustering


# Play with gamma to optimize the clustering results
kmns = SpectralClustering(n_clusters=2,  gamma=0.5, affinity='rbf', eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1, kernel_params=None, n_jobs=1)
kY = kmns.fit_predict(X)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)


ax1.scatter(Y[:,0],Y[:,1],  c=kY, cmap = "jet", edgecolor = "None", alpha=0.35)
ax1.set_title('Spectral clustering plot')

ax2.scatter(Y[:,0],Y[:,1],  c = datas['class'], cmap = "jet", edgecolor = "None", alpha=0.35)
ax2.set_title('Actual clusters')
plt.show()

#hiearchical clustering 
from sklearn.cluster import AgglomerativeClustering
aggC = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete')
kY = aggC.fit_predict(X)


f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)


ax1.scatter(Y[:,0],Y[:,1],  c=kY, cmap = "jet", edgecolor = "None", alpha=0.35)
ax1.set_title('Hierarchical clustering plot')

ax2.scatter(Y[:,0],Y[:,1],  c = datas['class'], cmap = "jet", edgecolor = "None", alpha=0.35)
ax2.set_title('Actual clusters')
plt.show()

#dendogram 
from scipy.cluster.hierarchy import dendrogram, linkage
Z = linkage(X)
dendrogram(Z)
plt.show()