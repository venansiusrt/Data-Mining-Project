import math
import numpy as np
import pandas as pd
from sklearn import model_selection
from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline 
plt.style.use('seaborn-whitegrid')

df = pd.read_csv('suicide.csv')
x = np.array()
x_train, y_train, x_test, y_test = model_selection.train_test_split(x,y,test_size=0.2)