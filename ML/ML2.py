import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
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
ordinal_encoder = OrdinalEncoder()
y_train = ordinal_encoder.fit_transform(y_train)
y_test = ordinal_encoder.fit_transform(y_test)
y_resampled = ordinal_encoder.fit_transform(y_resampled)

y_train=y_train.ravel()
y_test=y_test.ravel()
y_resampled =y_resampled .ravel()
#%%
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
# y_test = np.asarray(y_test)
# y_train = np.asarray(y_train)

#%%encoding the output
import tensorflow as  tf
input_shape = X_test.shape[1]
output_shape = y_train.shape[1]
model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(input_shape,)))
model.add(tf.keras.layers.Dense(320/2, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(352/2, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(160/2, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(32/2,  activation='relu'))
model.add(tf.keras.layers.Dense(output_shape, activation = 'softmax'))
model.summary()
# model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.compile(optimizer = 'adam', loss = 'Cross-entropy', metrics = ['accuracy'])
history = model.fit(x= X_train, y=y_train, batch_size = 32, epochs = 128, validation_data = (X_test, y_test))


#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
y_train = np.asarray(y_train)

# Create a RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# Wrap the RandomForestClassifier in a MultiOutputClassifier
clf = MultiOutputClassifier(rfc, n_jobs=-1)

# Fit the classifier to the data
clf.fit(X_train.values, y_train)

# Create a RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# Wrap the RandomForestClassifier in a MultiOutputClassifier
clf = MultiOutputClassifier(rfc, n_jobs=-1)

# Fit the classifier to the data
clf.fit(X, y)



#%%
from sklearn.datasets import make_multilabel_classification
import numpy as np
import xgboost as xgb

params = {
    'objective': 'multi:softprob',
    'num_class': 3,  # FCC, BCC, or both
    'max_depth': 6,
    'eta': 0.3,
    'gamma': 0.1,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'verbosity': 1,
    'n_jobs': -1,
    'seed': 42
}

# Train XGBoost classifier
model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train)















#%%
cat_num2 =  y_resampled['Phase'].value_counts()
ordinal_encoder = OrdinalEncoder()
y_train = ordinal_encoder.fit_transform(y_train)
y_test = ordinal_encoder.fit_transform(y_test)
y_resampled = ordinal_encoder.fit_transform(y_resampled)

y_train=y_train.ravel()
y_test=y_test.ravel()
y_resampled =y_resampled .ravel()

#%%