import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
df = pd.read_csv('6676_70_noP.csv')

# data = data.set_index('index')
output = df['Phase'].to_frame()
df.drop(df.columns[-1], axis=1, inplace=True) # drop phase
# df.drop(data.columns[1], axis=1, inplace=True)
# df.drop(data.columns[1], axis=1, inplace=True)
#%%
df = df.iloc[:,:-14]
df = pd.concat([df, output], axis=1)


#%%
cat_num = df['Phase'].value_counts()
el_num = df['Num_el'].value_counts()
df = df.iloc[:, 2:]
#%%
def phase_separation(df):
    BCC = df[df['Phase'] == 'BCC']
    FCC = df[df['Phase'] == 'FCC']
    BCC_FCC =  df[df['Phase'] == 'BCC+FCC']
    AM = df[df['Phase'] == 'AM']
    IM = df[df['Phase'] == 'IM']
    HCP= df[df['Phase'] == 'HCP']
    FCC_IM = df[df['Phase'] == 'FCC+IM']
    BCC_IM = df[df['Phase'] == 'BCC+IM']
    
    return BCC, FCC, HCP, BCC_FCC, AM, IM, BCC_IM, FCC_IM

def remove_categorical(df):#Find and remove all categorical columns
    columns = df.columns
    categorical = [col for col in columns if df[col].dtype == "object"]
    df = df.drop(categorical, axis=1)
    return df
#%%
BCC, FCC, HCP, BCC_FCC, AM, IM, BCC_IM, FCC_IM = phase_separation(df)
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
ax.set_ylabel('Compositon (mean value)',)
# ax.set_title('Presence of elements in BCC and FCC')
ax.legend()

fig.savefig('comp_el(FCC_BCC).jpg', dpi=600)

#%% Intermetallic vs Amorphous

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 20})
IM = remove_categorical(IM)
IM_ = IM.describe()
AM = remove_categorical(AM)
AM_ = AM.describe()


top_10_IM = IM_.loc['mean'].nlargest(12)
top_10_AM = AM_.loc['mean'].nlargest(12)


fig2, ax2 = plt.subplots(figsize=(12, 8))

ax2.bar(top_10_IM.index, top_10_IM, width=0.4,
       align='edge', label='Intermetallic', color='black')
    
ax2.bar(top_10_AM.index, top_10_AM, width=-0.4,
       align='edge', label='Amorphous', color = 'y')

ax2.set_ylabel('Compositon (mean value)',)

ax2.legend()

fig2.savefig('comp_el(AM_IM).jpg', dpi=600)

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
plot(FCC_IM, "FI")

plot(BCC_IM, "FI")
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
