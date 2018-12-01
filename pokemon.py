import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split	
from sklearn import preprocessing, cluster
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

pokemon = pd.read_csv('Pokemon.csv', index_col=0)
pokemon_mod = pokemon.fillna(" ")


print(pokemon_mod.head)

