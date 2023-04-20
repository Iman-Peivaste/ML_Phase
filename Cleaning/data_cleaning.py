import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
data = pd.read_csv('dataset10387_70.csv')
data.drop(data.columns[0], axis=1, inplace=True)
df = data.iloc[:, :-2].join(data.iloc[:, -1:].join(data.iloc[:, -2:-1]))
df.drop_duplicates(keep='first', inplace=True) # droping duplicates 
#%%
duplicates = df[df.iloc[:, :-1].duplicated(keep=False)]
grouped = duplicates.groupby(list(duplicates.iloc[:, :-1].columns))
mask = grouped['Phase'].nunique() > 1
indices_to_drop = mask[mask].index
df.drop(df.index[df.iloc[:, :-1].apply(tuple, axis=1).isin(indices_to_drop)], inplace=True)
#%%
# data = data.set_index('index')
output = df['Phase'].to_frame()
df.drop(df.columns[-1], axis=1, inplace=True) # drop phase
# df.drop(data.columns[1], axis=1, inplace=True)
# df.drop(data.columns[1], axis=1, inplace=True)
#%%
# df = df.iloc[:,:-14]
df = pd.concat([df, output], axis=1)
df.to_csv('6676_70_noP.csv', index=False)
