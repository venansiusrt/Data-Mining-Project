# Tugas Besar Data Mining 
## Dipelajari
- Classificaiton 
- Clustering 
- Sequence Pattern 
## Latar Terbelakang
![keren nih anjir](https://github.com/rasbt/pattern_classification/raw/master/Images/logo.png)

# Referensi 
https://github.com/rushter/MLAlgorithms

https://github.com/eriklindernoren/ML-From-Scratch

# TO DO 
- [ ] bikin time measurement per algoritma
- Classification 
    - [x] KNN
    - [x] [SVM](https://github.com/nalamidi/Breast-Cancer-Classification-with-Support-Vector-Machine/blob/master/Breast%20Cancer%20Classification.ipynb)
    - [ ] Random Forest
- Clustering 
    - [x] K means
    - [x] SpectralClustering
    - [x] Hierarchical Clustering
- Sequence Pattern
    - [x] Cari library
    - [ ] Prefix Span 
    - [ ] Generalized Sequence Pattern 
- [ ] Laporan ipnyb
- [ ] Exploratory Data Analysis

# Reference 
https://www.kaggle.com/rcfreitas/python-ml-breast-cancer-diagnostic-data-set


https://github.com/Jean-njoroge/Breast-cancer-risk-prediction

#### Multi Layer Perceptron 
Jadi biar lebih seru, urutan dari x_train dan x_Test di shuffle pake sklearn.utils.shuffle. Lalu dibandingkan dengan data asli
- 19 Epoch, 4 Layer (memakai adam optimiser, backprop)
![](https://i.imgur.com/segumWO.png)
31 23 17 13 (input_nodes: train_X.shape, hidden_nodes1: input_nodes, hidden_nodes2, hidden_nodes3)    
Epoch:  0   Accuracy:  0.62719   Cost:  303.86292   Valid Accuracy:  0.60714   Valid Cost:  37.30396   
Epoch:  1   Accuracy:  0.63158   Cost:  274.62439   Valid Accuracy:  0.60714   Valid Cost:  33.73276   
Epoch:  2   Accuracy:  0.69956   Cost:  224.27368   Valid Accuracy:  0.62500   Valid Cost:  27.30453   
Epoch:  3   Accuracy:  0.86842   Cost:  172.38452   Valid Accuracy:  0.87500   Valid Cost:  21.12439   
Epoch:  4   Accuracy:  0.93860   Cost:  138.36185   Valid Accuracy:  0.89286   Valid Cost:  17.99998   
Epoch:  5   Accuracy:  0.95614   Cost:  110.83379   Valid Accuracy:  0.91071   Valid Cost:  14.10146   
Epoch:  6   Accuracy:  0.96930   Cost:  77.26272   Valid Accuracy:  0.94643   Valid Cost:  8.75187   
Epoch:  7   Accuracy:  0.97149   Cost:  55.30008   Valid Accuracy:  0.98214   Valid Cost:  4.56376   
Epoch:  8   Accuracy:  0.97807   Cost:  41.37629   Valid Accuracy:  1.00000   Valid Cost:  2.64346   
Epoch:  9   Accuracy:  0.97588   Cost:  37.30238   Valid Accuracy:  1.00000   Valid Cost:  1.97846   
Epoch:  10   Accuracy:  0.97807   Cost:  34.33464   Valid Accuracy:  1.00000   Valid Cost:  1.60540   
Epoch:  11   Accuracy:  0.97588   Cost:  33.48156   Valid Accuracy:  1.00000   Valid Cost:  1.33480   
Epoch:  12   Accuracy:  0.97588   Cost:  29.47034   Valid Accuracy:  1.00000   Valid Cost:  1.19615   
Epoch:  13   Accuracy:  0.98684   Cost:  31.13475   Valid Accuracy:  1.00000   Valid Cost:  1.16577   
Epoch:  14   Accuracy:  0.98684   Cost:  26.50785   Valid Accuracy:  1.00000   Valid Cost:  1.20009   
Epoch:  15   Accuracy:  0.98904   Cost:  22.84942   Valid Accuracy:  1.00000   Valid Cost:  1.25582   
Epoch:  16   Accuracy:  0.98684   Cost:  25.87312   Valid Accuracy:  1.00000   Valid Cost:  1.25636   
Epoch:  17   Accuracy:  0.98684   Cost:  26.54008   Valid Accuracy:  1.00000   Valid Cost:  1.16558   
Epoch:  18   Accuracy:  0.98684   Cost:  25.67315   Valid Accuracy:  1.00000   Valid Cost:  1.10405   
Epoch:  19   Accuracy:  0.98684   Cost:  23.48993   Valid Accuracy:  1.00000   Valid Cost:  1.06180   
Run Complete Finished.
