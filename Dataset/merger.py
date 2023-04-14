#Merger
import pandas as pd
xl= pd.ExcelFile('Dataset_HEAs15.xlsx')
dfs = [xl.parse(sheet_name) for sheet_name in xl.sheet_names]
df = pd.concat(dfs, ignore_index=True)
#%%
df = df.drop(df[df.Phase == 'multi-phase'].index)
cat = df['Phase'].value_counts()
to_keep  = cat.index[:12]
df = df[df['Phase'].isin(to_keep)]
df.to_csv('1026_2_noP.csv', index=False)
#%%
cat = df['Phase'].value_counts()

df = df.drop_duplicates()
df.to_csv('8054_2_noP.csv', index=False)
