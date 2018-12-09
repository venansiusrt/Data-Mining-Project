# Classification 
menentukan kanker benign(pasif) / malign(aktif) terhadap umur dan gaji 
menggunakan f1 score untuk menentukan "nilai" klasifikasi
- k nearest neighbor (knn.py)
Test Accuracy:  0.9785714285714285
Train Accuracy:  0.9856887298747764 
  ![](https://i.imgur.com/zTZITgJ.png)   
  ![](https://i.imgur.com/yH2bD0A.png)
- SVM 
Test Accuracy:  0.9714285714285714
Train Accuracy:  0.9588550983899821
- Random Forest

## Dataset 
- [UCI Breast Cancer](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original))

   #  Attribute                     Domain
   -----------------------------------------
   1. Sample code number            id   number
   2. Clump Thickness               1 - 10
   3. Uniformity of Cell Size       1 - 10
   4. Uniformity of Cell Shape      1 - 10
   5. Marginal Adhesion             1 - 10
   6. Single Epithelial Cell Size   1 - 10
   7. Bare Nuclei                   1 - 10
   8. Bland Chromatin               1 - 10
   9. Normal Nucleoli               1 - 10
  1.  Mitoses                       1 - 10
  2.  Class:                        (2 for benign, 4 for malignant)