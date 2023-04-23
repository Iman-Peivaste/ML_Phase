#this is for anomoly detection

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.ensemble import IsolationForest
#%% Read input
df = pd.read_csv('6676_70_noP.csv')
df['Omega'] = df['Melting_point_K'] * df['entropy'] / abs(df['Enthalpy'])
df = df.iloc[:,-16:]

df = df.drop('Geo', axis=1)
df = df.drop('Omega', axis=1)
output = df['Phase'].to_frame()
cat_list = output.index.tolist()
cat_num =  df['Phase'].value_counts()
#%%
def remove_outliers(data):
    data_encoded = pd.get_dummies(data, columns=['Phase'])

    model = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
    # Fit the model to the data
    model.fit(data_encoded)
    
    # Predict the outliers
    outliers = model.predict(data_encoded)
    
    # Filter the outliers from the original dataset
    data_filtered = data[~(outliers == -1)]
    cat_vars = ['Phase']
    filtered_cat_vars = data_filtered[cat_vars]
    filtered_data = pd.concat([data_filtered.drop(cat_vars, axis=1), filtered_cat_vars], axis=1)
    return filtered_data

#%%
# df = remove_outliers(df)

#%%
#remove based on each category
BCC = df[df['Phase'] == 'BCC']
FCC = df[df['Phase'] == 'FCC']
FCC_BCC =  df[df['Phase'] == 'BCC+FCC']

BCC_IM = df[df['Phase'] == 'BCC+IM']
FCC_IM = df[df['Phase'] == 'FCC+IM']
FCC_BCC_IM =  df[df['Phase'] == 'BCC+FCC+IM']
IM =  df[df['Phase'] == 'IM']

AM =  df[df['Phase'] == 'AM']
BCC_AM =  df[df['Phase'] == 'BCC+AM']
FCC_AM =  df[df['Phase'] == 'FCC+AM']
FCC_BCC_AM =  df[df['Phase'] == 'BCC+FCC+AM']

dfs = [BCC, FCC, FCC_BCC, BCC_IM, FCC_IM, FCC_BCC_IM,IM, AM, BCC_AM,FCC_AM , FCC_BCC_AM ]

new_df = []
for dd in dfs:
    dd = remove_outliers(dd)
    new_df.append(dd)
df = pd.concat(new_df, axis=0, ignore_index=True)

#%% Vis BCC, FCC, BCC+FCC. Vec delta
'''
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 20})

def plt_vec_size(df):
    fig, ax = plt.subplots(figsize=(12, 8))
    categories = df['Phase']
    
    # define a color map for the categories
    colors = {'FCC': 'r', 'BCC': 'g', 'BCC+FCC': 'b'}
    color_list = [colors[cat] for cat in categories]
    
    plt.scatter(x = df['VEC'], y=df['Pauling_EN'], c =color_list)
    # ax.scatter(x = df['VEC'], y=df['Atomic_radius_calculated_dif'], c =color_list)
    
    
    handles = [plt.plot([],[], marker="o", ls="", color=color, label=label)[0] 
               for label, color in colors.items()]
    plt.legend(handles=handles, title='Categories', loc='lower right')
    
    
    plt.xlabel('VEC')
    plt.ylabel((r'$Electronegativity$'))
    fig.savefig('Pauling_EN(FCC&BCC)_NOaut.jpg', dpi=600)


BCC = df[df['Phase'] == 'BCC']
BCC_ = BCC.describe()
FCC = df[df['Phase'] == 'FCC']
FCC_ = FCC.describe()
FCC_BCC =  df[df['Phase'] == 'BCC+FCC']
FCC_BCC_= FCC_BCC.describe()

dfs = [BCC, FCC, FCC_BCC]
df = pd.concat(dfs, axis=0, ignore_index=True)

plt_vec_size(df)
'''
#%%
cats_num = df['Phase'].value_counts()
df.to_csv('5677_14.csv', index=False)
