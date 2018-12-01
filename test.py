import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
from matplotlib import rcParams


# Defining colors for graphs 
colors = {
    "Bug": "#A6B91A",
    "Dark": "#705746",
    "Dragon": "#6F35FC",
    "Electric": "#F7D02C",
    "Fairy": "#D685AD",
    "Fighting": "#C22E28",
    "Fire": "#EE8130",
    "Flying": "#A98FF3",
    "Ghost": "#735797",
    "Grass": "#7AC74C",
    "Ground": "#E2BF65",
    "Ice": "#96D9D6",
    "Normal": "#A8A77A",
    "Poison": "#A33EA1",
    "Psychic": "#F95587",
    "Rock": "#B6A136",
    "Steel": "#B7B7CE",
    "Water": "#6390F0",
}


# robberies = pd.read_csv('robberies.csv')
# mod_robberies = robberies.fillna(" ")
# check = mod_robberies.dtypes

# df = pd.read_csv('datatilang.csv')
# mod_df = df.fillna(" ")
# command = mod_df.describe()



pokemon = pd.read_csv('Pokemon.csv', index_col=0)
pokemon_mod = pokemon.fillna(" ")

print(pokemon_mod.columns[1])
# pokemon_mod2 = pokemon_mod.dropna("#")

