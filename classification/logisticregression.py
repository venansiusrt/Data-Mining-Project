import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV

dataset = pd.read_csv('breast-cancer-wisconsin.data', index_col=0)
dataset.replace('?', '1', inplace=True) #-9999 biar outlier, gak masuk ke grafik 
X = dataset.drop(['class'], axis = 1) # We drop our "target" feature and use all the remaining features in our dataframe to train the model.
print(X.head())
y = dataset['class']
print('\n')
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

lgr = LogisticRegression()



# standardizing and PCA
scaler = StandardScaler()
P_scaled=scaler.fit_transform(X)
P_scaled=pd.DataFrame(P_scaled)

pca=PCA(n_components=0.95)
P_pca=pca.fit_transform(P_scaled)
print (P_pca.shape)
print(pca.explained_variance_ratio_) 
print (pca.explained_variance_ratio_.sum())

n=['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','diagnosis']
#merging the reduced data with diagnosis column

# plotting the the first 2 pca components against diagnosis
sns.lmplot("PC1", "PC2", hue="class", data=X, fit_reg=False,markers=["o", "x"],palette="Set1")
sns.plt.show()

# param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
# scores = ['accuracy', 'recall']
# for sc in scores:
#     grid_lgr=GridSearchCV(lgr,param_grid,cv=10,scoring=sc,n_jobs=-1)
#     print("# Tuning hyper-parameters for %s" % sc)
#     grid_lgr.fit(X_train,y_train)
#     print(grid_lgr.best_params_)
#     print(np.round(grid_lgr.best_score_,3))
