import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score
import tensorflow as  tf
import matplotlib.pyplot as plt
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.multitask import TabNetMultiTaskClassifier

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
from imblearn.pipeline import make_pipeline
from collections import Counter
#%%
df = pd.read_csv('5677_14.csv')
cat_num =  df['Phase'].value_counts()




#%%
'''
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
'''
'''
dfs = [BCC, FCC, FCC_BCC, BCC_IM, FCC_IM, FCC_BCC_IM,IM, AM, BCC_AM,FCC_AM , FCC_BCC_AM ]

df2 = pd.concat(dfs, axis=0, ignore_index=True)
'''
#%%

BCC = df[df['Phase'] == 'BCC']
FCC = df[df['Phase'] == 'FCC']
FCC_BCC =  df[df['Phase'] == 'BCC+FCC']
dfs = [BCC, FCC, FCC_BCC]

df = pd.concat(dfs, axis=0, ignore_index=True)
#%%
def normalization (df_in, method='std',  bb=0, ee=13):
    
    need_scale = df_in.columns[bb:ee]
    #need_scale = df_in.columns[26:35]
    
    # here we need to selcet which type of standardization wee need
    if method == 'std':
        scaler = StandardScaler()
    else: 
        scaler = MinMaxScaler()
    
    cols_to_norm = need_scale
    df_in[cols_to_norm] = scaler.fit_transform(df_in[cols_to_norm])
    return df_in

def input_output(df, output_name= 'Phase', cat_name = 'process'):
    # cat_encoder = OneHotEncoder()
    ordinal_encoder = OrdinalEncoder()
    df_out = ordinal_encoder.fit_transform(df[[output_name]])
    df_out = df_out.ravel()
    df_in = df.drop(output_name, axis=1)
    # df_in = df_in.drop('Alloy ', axis=1)
    # df_encoded = pd.get_dummies(df_in[cat_name], prefix=cat_name)
    # df_in = pd.concat([df_in.drop(cat_name, axis=1), df_encoded], axis=1)
    return df_in, df_out
def split(df_in, df_out, test_size = 0.1 ):
    X_train, X_test, y_train, y_test = train_test_split(df_in, df_out, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test
def RF_train(X_train, y_train,n_estimators ):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    print('Begining of fitting')
    model.fit(X_train, y_train)
    print('End of fitting')
    return model
#%%
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.ensemble import RandomForestClassifier
seed = 66
df = normalization(df )

output = df['Phase'].to_frame()
df.drop(df.columns[-1], axis=1, inplace=True)
# inp, out = input_output(df, 'Phase', 'process')
X_train, X_test, y_train, y_test =  split(df, output)
# X_train, X_test, y_train, y_test =  split(inp, out)

# X_train, y_train = ADASYN().fit_resample(X_train, y_train)
X_resampled, y_resampled = ADASYN().fit_resample(X_train, y_train)


X_train.to_csv('X_train.csv', index=False)
y_train.to_csv('y_train.csv', index=False)

X_test.to_csv('X_test.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

X_resampled.to_csv('X_resampled.csv', index=False)
y_resampled.to_csv('y_resampled.csv', index=False)





cat_num2 =  y_resampled['Phase'].value_counts()
#%%
plt.figure()
ax = cat_num2.plot.bar(color='blue', label='Synthesized data')
cat_num.plot.bar(color='red', label='Original data', ax=ax)
plt.subplots_adjust(bottom=0.4)
plt.legend(loc='upper right', bbox_to_anchor=(1, 1.21))
plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 9})
plt.ylabel("Number of alloys")
plt.show()
plt.savefig('Cat_caunt.jpg', dpi=600, bbox_inches='tight')
#%%
print(sorted(Counter(y_resampled,).items()))



rf = RandomForestClassifier(random_state=seed)
rf.fit(X_resampled, y_resampled).score(X_test, y_test)
model = make_pipeline(
    SMOTEENN(random_state=seed),
    RandomForestClassifier(random_state=seed)
)

cv_results = cross_validate(
    estimator=model, 
    X=X_resampled, 
    y=y_resampled, 
    return_train_score=True, 
    return_estimator=True,
    n_jobs=-1,
    cv=10
)

print(
    f"Accuracy mean +/- std. dev.: "
    f"{cv_results['test_score'].mean():.2f} +/- "
    f"{cv_results['test_score'].std():.2f}"
)



#%%
model = RF_train(X_train, y_train,n_estimators=1000 )
predictions = model.predict(X_test)
lst_column=X_train.columns.tolist()
print("Accuracy:", accuracy_score(y_test, predictions))
importances = model.feature_importances_
print(importances)

imp_list = importances.tolist()
imp= pd.DataFrame(importances.reshape(-1,13), columns=lst_column)
imp = imp.T.sort_values(by=0)

# fig, ax = plt.subplots(figsize=(12, 8))
x = imp[-6:]
x = x.rename(index={'Atomic_radius_calculated_dif': 'Atomic size difference', 
                    'DFT_LDA_Etot': 'Total Energy (DFT)',
                    'no_of_valence_electrons': 'NVE',
                    'Enthalpy': 'Mixing Enthalpy',
                    'Pauling_EN': 'Electronegarivity'})
ax = x.plot(kind='bar')
ax.legend().remove()
plt.subplots_adjust(bottom=0.45)

plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 16})
plt.savefig('imp_RF(BCC_FCC).jpg', dpi=300, bbox_inches='tight')
# x.plot(kind='bar')


