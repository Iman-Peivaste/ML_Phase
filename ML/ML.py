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
from tqdm import tqdm
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
from imblearn.pipeline import make_pipeline
from collections import Counter
import xgboost as xgb

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA

from sklearn.feature_extraction import DictVectorizer
import shap
#%%
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')
X_resampled = pd.read_csv('X_resampled.csv')
y_resampled = pd.read_csv('y_resampled.csv')

X_train = X_resampled
y_train = y_resampled


cat_num =  y_resampled['Phase'].value_counts()
#%%
keep_categories = ["BCC", "FCC", "BCC+FCC"]
X_train = X_train[y_train["Phase"].isin(keep_categories)]

# filter the training output data
y_train = y_train[y_train["Phase"].isin(keep_categories)]

# filter the testing input data
X_test = X_test[y_test["Phase"].isin(keep_categories)]

# filter the testing output data
y_test = y_test[y_test["Phase"].isin(keep_categories)]


#%%encoding the output

cat_num2 =  y_resampled['Phase'].value_counts()
ordinal_encoder = OrdinalEncoder()
y_train = ordinal_encoder.fit_transform(y_train)
y_test = ordinal_encoder.fit_transform(y_test)
y_resampled = ordinal_encoder.fit_transform(y_resampled)

y_train=y_train.ravel()
y_test=y_test.ravel()
y_resampled =y_resampled .ravel()

#%%Random Forest
def RF_train(X_train, y_train,n_estimators ):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    print('Begining of fitting')
    model.fit(X_train, y_train)
    print('End of fitting')
    return model

model = RF_train(X_train, y_train,n_estimators=1000 )
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
lst_column=X_train.columns.tolist()
importances = model.feature_importances_
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


#%%
clf = TabNetClassifier()  #TabNetRegressor()
clf.fit(
  X_resampled, y_resampled,
  eval_set=[(X_test, y_test)]
)
preds = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))
#%%
# #%%
# learning_rate_range = np.arange(0.01, 1, 0.05)
# test_XG = [] 
# train_XG = []
# for lr in tqdm(learning_rate_range):
#     xgb_classifier = xgb.XGBClassifier(eta = lr)
#     # xgb_classifier.fit(X_resampled, y_resampled)
#     xgb_classifier.fit(X_train, y_train)
#     train_XG.append(xgb_classifier.score(X_train, y_train))
#     # train_XG.append(xgb_classifier.score(X_resampled, y_resampled))
#     test_XG.append(xgb_classifier.score(X_test, y_test))
 #%%
from xgboost import plot_importance
xgb_classifier = xgb.XGBClassifier(eta = 0.16) 
xgb_classifier.fit(X_train, y_train) 
print("Accuracy:", xgb_classifier.score(X_test, y_test))

fig, ax = plt.subplots(figsize=(10, 8))
plot_importance(xgb_classifier, ax=ax)
plt.show()


# vec = DictVectorizer()
# booster = xgb.get_booster()
# original_feature_names = booster.feature_names
# booster.feature_names = vec.get_feature_names()
# print(booster.get_dump()[0])
# # recover original feature names
# booster.feature_names = original_feature_name


#%%
# fig = plt.figure(figsize=(10, 7))
# plt.plot(learning_rate_range, train_XG, c='orange', label='Train')
# plt.plot(learning_rate_range, test_XG, c='m', label='Test')
# plt.xlabel('Learning rate')
# plt.xticks(learning_rate_range)
# plt.ylabel('Accuracy score')
# plt.ylim(0.6, 1)
# plt.legend(prop={'size': 12}, loc=3)
# plt.title('Accuracy score vs. Learning rate of XGBoost', size=14)
# plt.show()
#%%
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')
X_resampled = pd.read_csv('X_resampled.csv')
y_resampled = pd.read_csv('y_resampled.csv')

X_train = X_resampled
y_train = y_resampled


cat_num =  y_resampled['Phase'].value_counts()
#%%
keep_categories = ["BCC", "FCC", "BCC+FCC"]
X_train = X_train[y_train["Phase"].isin(keep_categories)]

# filter the training output data
y_train = y_train[y_train["Phase"].isin(keep_categories)]

# filter the testing input data
X_test = X_test[y_test["Phase"].isin(keep_categories)]

# filter the testing output data
y_test = y_test[y_test["Phase"].isin(keep_categories)]
#%%
y_train = y_train.values.reshape(-1,1)
y_test = y_test.values.reshape(-1,1)

cat_encoder = OneHotEncoder()
y_train = cat_encoder.fit_transform(y_train)
y_train = y_train.toarray()

y_test = cat_encoder.fit_transform(y_test)
y_test = y_test.toarray()
#%%

#%%
history = model.fit(x= X_train.values, y=y_train, batch_size = 32, epochs = 64, validation_data = (X_test.values, y_test))
background = X_train.values[np.random.choice(X_train.values.shape[0], 100, replace=False)]
explainer = shap.KernelExplainer(model.predict, background)
shap_values = explainer.shap_values(X_test.values)

# Plot feature importance



fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, feature_names=X_train.columns, plot_type='bar', plot_size=(10,8))
plt.show()

#%%
csfont = {'fontname':'Times New Roman'}
plt.rcParams["font.family"] = "Times New Roman"
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.style.use('default')

epochs = range(1, len(loss) + 1)
csfont = {'fontname':'Times New Roman'}
plt.style.use('default')
plt.figure(figsize=(12, 6))

plt.plot(epochs, val_loss, 'r', label='Validation',linewidth = 3)
plt.plot(epochs, loss, 'y', label='Training', linewidth = 3)
#%%
acc = history.history['accuracy']
#acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.style.use('default')
plt.figure(figsize=(12, 6))
plt.plot(epochs, acc, 'y', label='Training acc', linewidth = 2)
plt.plot(epochs, val_acc, 'r', label='Validation acc', linewidth = 2)
#plt.title('Training and validation accuracy',**csfont, fontweight='bold',fontsize=13)
plt.xlabel('Epochs', **csfont, fontweight='bold',fontsize=16)
plt.ylabel('Accuracy', **csfont, fontweight='bold',fontsize=16)
plt.rcParams['font.size'] = '16'

plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.grid()

#%%
from sklearn.inspection import permutation_importance
svm_clf = SVC(kernel="poly", degree=6, coef0=1, C=5)
svm_clf.fit(X_train, y_train) 
print("Accuracy:", svm_clf.score(X_test, y_test))

result = permutation_importance(svm_clf, X_train, y_train, n_repeats=10, random_state=42)
sorted_idx = result.importances_mean.argsort()[::-1]
labels = X_train.columns[sorted_idx]

# Plot feature importance
fig, ax = plt.subplots(figsize=(10, 8))
ax.boxplot(result.importances[sorted_idx].T, vert=False, labels=labels)
ax.set_title("Permutation Importances (train set)")
plt.tight_layout()
plt.show()


#%%

