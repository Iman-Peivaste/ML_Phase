#Feature corrolation 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
#%%
df = pd.read_csv('6676_70_noP.csv')
df['Omega'] = df['Melting_point_K'] * df['entropy'] / abs(df['Enthalpy'])
df = df.iloc[:,-16:]

output = df['Phase'].to_frame()
# df.drop(df.columns[-2], axis=1, inplace=True)

#%%
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 20})

def plt_vec_size(df):
    fig, ax = plt.subplots(figsize=(12, 8))
    categories = df['Phase']
    
    # define a color map for the categories
    colors = {'FCC': 'r', 'BCC': 'g', 'BCC+FCC': 'b'}
    color_list = [colors[cat] for cat in categories]
    
    # plt.scatter(x = df['VEC'], y=df['Atomic_radius_calculated_dif'], c =color_list)
    ax.scatter(x = df['VEC'], y=df['Atomic_radius_calculated_dif'], c =color_list)
    
    
    handles = [plt.plot([],[], marker="o", ls="", color=color, label=label)[0] 
               for label, color in colors.items()]
    plt.legend(handles=handles, title='Categories', loc='upper right')
    
    
    plt.xlabel('VEC')
    plt.ylabel((r'$\delta r$'))
    fig.savefig('vec_delta_r.jpg', dpi=600)


BCC = df[df['Phase'] == 'BCC']
BCC_ = BCC.describe()
FCC = df[df['Phase'] == 'FCC']
FCC_ = FCC.describe()
FCC_BCC =  df[df['Phase'] == 'BCC+FCC']
FCC_BCC_= FCC_BCC.describe()

dfs = [BCC, FCC, FCC_BCC]
df = pd.concat(dfs, axis=0, ignore_index=True)

plt_vec_size(df)
#%%
def plt_vec_size(df):
    fig, ax = plt.subplots(figsize=(12, 8))
    categories = df['Phase']
    
    # define a color map for the categories
    colors = {'FCC': 'r', 'BCC': 'g', 'BCC+FCC': 'b'}
    color_list = [colors[cat] for cat in categories]
    
    # plt.scatter(x = df['VEC'], y=df['Atomic_radius_calculated_dif'], c =color_list)
    ax.scatter(x = df['VEC'], y=df['Pauling_EN'], c =color_list)
    
    
    handles = [plt.plot([],[], marker="o", ls="", color=color, label=label)[0] 
               for label, color in colors.items()]
    plt.legend(handles=handles, title='Categories', loc='lower right')
    
    
    plt.xlabel('VEC')
    plt.ylabel(('Pauling_EN'))
    fig.savefig('vec_Pauling.jpg', dpi=600)


BCC = df[df['Phase'] == 'BCC']
BCC_ = BCC.describe()
FCC = df[df['Phase'] == 'FCC']
FCC_ = FCC.describe()
FCC_BCC =  df[df['Phase'] == 'BCC+FCC']
FCC_BCC_= FCC_BCC.describe()

dfs = [BCC, FCC, FCC_BCC]
df = pd.concat(dfs, axis=0, ignore_index=True)

plt_vec_size(df)

#%%
#BCC_IM VS FCC_IM

def plt_vec_size(df):
    fig, ax = plt.subplots(figsize=(12, 8))
    categories = df['Phase']
    
    # define a color map for the categories
    colors = {'IM': 'black','FCC+IM': 'r', 'BCC+IM': 'g'}
    color_list = [colors[cat] for cat in categories]
    
    # plt.scatter(x = df['VEC'], y=df['Atomic_radius_calculated_dif'], c =color_list)
    ax.scatter(x = df['VEC'], y=df['Atomic_radius_calculated_dif'], c =color_list)
    
    
    handles = [plt.plot([],[], marker="o", ls="", color=color, label=label)[0] 
               for label, color in colors.items()]
    plt.legend(handles=handles, title='Categories', loc='lower right')
    
    
    plt.xlabel('VEC')
    plt.ylabel((r'$\delta r$'))
    fig.savefig('vec_Pauling.jpg', dpi=600)


BCC_IM = df[df['Phase'] == 'BCC+IM']
BCC_IM_ = BCC_IM.describe()
FCC_IM = df[df['Phase'] == 'FCC+IM']
FCC_IM_ = FCC_IM.describe()
IM =  df[df['Phase'] == 'IM']
IM_= IM.describe()

dfs = [BCC_IM, FCC_IM, IM]
df = pd.concat(dfs, axis=0, ignore_index=True)

plt_vec_size(df)


#%%

def plt_vec_size(df):
    fig, ax = plt.subplots(figsize=(12, 8))
    categories = df['Phase']
    
    # define a color map for the categories
    colors = {'IM': 'black','FCC+IM': 'r', 'BCC+IM': 'g'}
    color_list = [colors[cat] for cat in categories]
    
    # plt.scatter(x = df['VEC'], y=df['Atomic_radius_calculated_dif'], c =color_list)
    ax.scatter(x = df['VEC'], y=df['Atomic_radius_calculated_dif'], c =color_list)
    
    
    handles = [plt.plot([],[], marker="o", ls="", color=color, label=label)[0] 
               for label, color in colors.items()]
    plt.legend(handles=handles, title='Categories', loc='lower right')
    
    
    plt.xlabel('VEC')
    plt.ylabel((r'$\delta r$'))
    fig.savefig('vec_Pauling.jpg', dpi=600)


BCC_IM = df[df['Phase'] == 'BCC+IM']
BCC_IM_ = BCC_IM.describe()
FCC_IM = df[df['Phase'] == 'FCC+IM']
FCC_IM_ = FCC_IM.describe()
IM =  df[df['Phase'] == 'IM']
IM_= IM.describe()

BCC_AM = df[df['Phase'] == 'BCC+AM']
BCC_AM_ = BCC_AM.describe()
FCC_AM = df[df['Phase'] == 'FCC+AM']
FCC_AM_ = FCC_AM.describe()
AM =  df[df['Phase'] == 'AM']
AM_= AM.describe()

dfs = [BCC_AM, FCC_AM, AM]
df = pd.concat(dfs, axis=0, ignore_index=True)

plt_vec_size(df)


















#%%

df = df[df['Enthalpy'] != 0]
df = df[df['Atomic_radius_calculated'] != 0]

#%%

x=df
minn = []
maxx = []
for column in x.columns:
    minn.append(x[column].min())
    maxx.append(x[column].max())
    
minn = np.array(minn)
maxx = np.array(maxx)
np.save('min.npy', minn)
np.save('max.npy' , maxx)


for column in x.columns:
    x[column] = (x[column]-x[column].min())/(x[column].max()-x[column].min())






