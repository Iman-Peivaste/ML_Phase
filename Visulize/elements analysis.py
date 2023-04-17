import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
data = pd.read_csv('dataset10387_70.csv')
data.drop(data.columns[0], axis=1, inplace=True)
#%%
el_num = data['Num_el'].value_counts()
#%%
# data = data.set_index('index')
output = data['Phase'].to_frame()
data.drop(data.columns[-2], axis=1, inplace=True)
data.drop(data.columns[1], axis=1, inplace=True)
#%%
data = data.iloc[:,:-14]
df = pd.concat([data, output], axis=1)
df.drop(df.columns[0], axis=1, inplace=True)
cat_num = df['Phase'].value_counts()

df.drop_duplicates(keep='first', inplace=True)
#%%
def phase_separation(df):
    BCC = df[df['Phase'] == 'BCC']
    FCC = df[df['Phase'] == 'FCC']
    BCC_FCC =  df[df['Phase'] == 'BCC+FCC']
    AM = df[df['Phase'] == 'AM']
    IM = df[df['Phase'] == 'IM']
    HCP= df[df['Phase'] == 'HCP']
    # FCC_IM = df[df['Phase'] == 'FCC + Im']
    # BCC_IM = df[df['Phase'] == 'BCC + Im']
    
    
    # FCC_des = FCC.describe()
    # AM_des = AM.describe()
    
    # BCC_des = BCC.describe()
    # IM_des = IM.describe()
    # FCC_IM_des = FCC_IM.describe()
    # BCC_IM_des = BCC_IM.describe()
    # FCC_BCC_des = FCC_BCC.describe()
    
    return BCC, FCC, HCP, BCC_FCC, AM, IM

def remove_categorical(df):#Find and remove all categorical columns
    # find all columns in the dataframe
    columns = df.columns
    # check if each column is categorical
    categorical = [col for col in columns if df[col].dtype == "object"]
    # drop the categorical columns from the dataframe
    df = df.drop(categorical, axis=1)
    return df
#%%
BCC, FCC, HCP, BCC_FCC, AM, IM = phase_separation(df)
#%% FCC versus BCC

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 20})
BCC = remove_categorical(BCC)
BCC_ = BCC.describe()
FCC = remove_categorical(FCC)
FCC_ = FCC.describe()
IM = remove_categorical(IM)
IM_ = IM.describe()

top_10_BCC = BCC_.loc['mean'].nlargest(11)
top_10_FCC = FCC_.loc['mean'].nlargest(11)

fig, ax = plt.subplots(figsize=(12, 8))
# col_names_row1 = ['Cr', 'Al']
# top_10_BCC.index = col_names_row1
# top_10_FCC.index = col_names_row1
ax.bar(top_10_BCC.index, top_10_BCC, width=0.4,
       align='edge', label='BCC', color='b')
    
ax.bar(top_10_FCC.index, top_10_FCC, width=-0.4,
       align='edge', label='FCC', color = 'r')
ax.set_ylabel('Compositon (mean Value)',)
# ax.set_title('Presence of elements in BCC and FCC')
ax.legend()

fig.savefig('comp_el(FCC_BCC).jpg', dpi=600)

#%%

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 20})
BCC = remove_categorical(BCC)
BCC_ = BCC.describe()
FCC = remove_categorical(FCC)
FCC_ = FCC.describe()
IM = remove_categorical(IM)
IM_ = IM.describe()

top_10_BCC = BCC_.loc['mean'].nlargest(11)
top_10_FCC = FCC_.loc['mean'].nlargest(11)
top_10_IM = IM_ .loc['mean'].nlargest(11)

fig, ax = plt.subplots(figsize=(12, 8))
# col_names_row1 = ['Cr', 'Al']
# top_10_BCC.index = col_names_row1
# top_10_FCC.index = col_names_row1
ax.bar(top_10_BCC.index, top_10_BCC, width=0.4,
       align='edge', label='BCC', color='b')
    
ax.bar(top_10_FCC.index, top_10_FCC, width=-0.4,
       align='edge', label='FCC', color = 'r')

ax.bar(top_10_IM.index, top_10_IM, width=-0.8,
       align='edge', label='IM', color = 'black')


ax.set_ylabel('Compositon (mean Value)',)
# ax.set_title('Presence of elements in BCC and FCC')
ax.legend()

fig.savefig('comp_el(FCC_BCC_IM).jpg', dpi=600)




#%%
def plot (xp, xpn):
    name = xpn
    x= remove_categorical(xp)
    x_ = x.describe()
    top = x_.loc['mean'].nlargest(10)
    axe = top.plot(kind='bar')

    axe.set_ylabel('The normalized mean Value compositoin')
    axe.set_title(f'{name}')
    
#%%
plot(IM, "IM")



#%%
BCC = remove_categorical(BCC)
BCC_ = BCC.describe()
# BCC.hist()
top_10_BCC = BCC_.loc['mean'].nlargest(10)
ax_BCC = top_10_BCC.plot(kind='bar')
ax_BCC.set_ylabel('The normalized mean Value')
ax_BCC.set_title('BCC')
#%%
FCC = remove_categorical(FCC)
FCC_ = FCC.describe()
top_10_FCC = FCC_.loc['mean'].nlargest(10)
ax_FCC=top_10_FCC.plot(kind='bar')
ax_FCC.set_ylabel('The normalized mean Value')
ax_FCC.set_title('FCC')
#%%
HCP = remove_categorical(HCP)
HCP_ = HCP.describe()
# BCC.hist()
top_10_HCP = HCP_.loc['mean'].nlargest(10)
ax_HCP = top_10_BCC.plot(kind='bar')
ax_HCP.set_ylabel('The normalized mean Value')
ax_HCP.set_title('HCP')
#%%
HCP = remove_categorical(IM)
HCP_ = HCP.describe()
# BCC.hist()
top_10_HCP = HCP_.loc['mean'].nlargest(10)
ax_HCP = top_10_BCC.plot(kind='bar')
ax_HCP.set_ylabel('The normalized mean Value')
ax_HCP.set_title('HCP')
