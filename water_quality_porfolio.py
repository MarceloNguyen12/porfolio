#!/usr/bin/env python
# coding: utf-8

# In[86]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import warnings
from sklearn.exceptions import FitFailedWarning
warnings.filterwarnings("ignore", category=FitFailedWarning)


from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBClassifier
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import BaggingClassifier



from sklearn.ensemble import RandomForestClassifier
# Set display options to show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Your code here

# Reset display options to default
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')
pd.reset_option('display.width')
pd.reset_option('display.max_colwidth')


# In[2]:


df = pd.read_csv(r"C:\Users\Admin\Downloads\archive (24)\water_potability.csv")


# In[3]:


df


# # Checking for missing values
# 

# In[4]:


df.info()


# In[5]:


print("Missing total")
print(df.isna().sum())
print('---------')
print('Missing percentages')
print(df.isna().sum() / df.shape[0])
print('---------')
missing = []
columns = df.columns
for i, v in enumerate(df.isna().sum()):
    if v > 0:
        missing.append(columns[i])
print(missing)


# In[6]:


df.columns


# # Dealing with missing data using KNN imputer 
# 

# In[7]:


X = df.drop('Potability', axis=1)
imputer = KNNImputer()
X_imputed = imputer.fit_transform(X)
df_ = pd.DataFrame(X_imputed, columns=X.columns)


# In[8]:


y = df['Potability']
df = pd.concat([df_, y], axis=1)
df


# # Perform basic EDA

# # 1. Do the histogram to check the distribution and the relationship between all the feature with the target, using hue too
# 
# - After plotting the histogram, i have seen that there are normal distribution of the features, however, there are still slight skewness in some features
# 
# - But i can see imbalance in the data where the number of class 0 is slightly larger than class 1 
# 

# In[9]:


for col in df.columns[:-1]:
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df,x=col, hue='Potability', kde=True, alpha=0.7)
    plt.show()
    


# 2. Checking for skewness 
# The skewness is not drastically, so I will leave it there without transformation process

# In[10]:


print('Feature with high skewness:')
for i, v in enumerate(df.skew()):
    if v > 0.5 or v < -0.5:
        print(columns[i], ':', v)


# # 3. Checking the correlation and relationship between features and the target using pair plot and heatmap 
# 
# 
# # Though there are low correlation between the features and the target, and no clear relationship 

# In[11]:


sns.pairplot(df)


# In[12]:


correlation = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation, annot=True, lw=1)


# # 3. Creating new features 
# - My strategy is to base on the information given from the dataset of the thresholds of each chemical element that affect the Potability of water 
# 

# In[13]:


df.columns


# In[14]:


df['ph_level'] = pd.cut(df['ph'], bins=[0, 6.9, 7.1, 14], labels=['acidic', 'neutral', 'alkaline'], include_lowest=True)
df['tds_level'] = pd.cut(df['Solids'], bins=[0, 500, 1000, float('inf')], labels=['low', 'moderate', 'high'], include_lowest=True)
df['chloramines_level'] = pd.cut(df['Chloramines'],  bins=[0, 4, float('inf')], labels=['low', 'high'], include_lowest=True)
df['trihalomethanes_level'] = pd.cut(df['Trihalomethanes'], bins=[0, 80, float('inf')], labels=['safe', 'unsafe'], include_lowest=True)

ordinal_mapping = {
    'acidic': 0,
    'neutral': 1,
    'alkaline': 2,
    'low': 0,
    'moderate': 1,
    'high': 2,
    'safe': 0,
    'unsafe': 1,
    'low': 0,
    'high': 1
}

df['ph_level'] = df['ph_level'].map(ordinal_mapping)
df['tds_level'] = df['tds_level'].map(ordinal_mapping)
df['chloramines_level'] = df['chloramines_level'].map(ordinal_mapping)
df['trihalomethanes_level'] = df['trihalomethanes_level'].map(ordinal_mapping)


# In[15]:


df


# In[16]:


y 


# In[17]:


df


# In[18]:


correlation = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation, annot=True, lw=1)


# # Dealing with imbalanced data in the target 
# So i did try to use a baseline models to  check if fixing imbalanced target can improve the prediction, in this case i use logistic but it seems that i didn't have much improvements, so i will still use the balanced target 
# 

# In[19]:


df['Potability'].value_counts()


# In[20]:


X, y = df.drop('Potability', axis=1), df['Potability']
oversampler = SMOTE(sampling_strategy='auto', random_state=42)
undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)

X_resampled, y_resampled = oversampler.fit_resample(X, y)
X_resampled, y_resampled = undersampler.fit_resample(X_resampled, y_resampled)
print(pd.Series(y_resampled).value_counts())


# In[21]:


df.Potability.value_counts()


# In[22]:


df_copy = df.copy()
df_copy['Potability'] = y_resampled


# In[23]:


df_copy


# In[24]:


correlation_after_resample = df_copy.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_after_resample, annot=True, lw=1)


# In[25]:


X, y = df.drop('Potability', axis=1), df['Potability']
X_copy, y_copy = df_copy.drop('Potability', axis=1), df_copy['Potability']
X_train, X_test, y_train, y_test =train_test_split(X,y, test_size=0.2, random_state=42)
X_train_copy, X_test_copy, y_train_copy, y_test_copy = train_test_split(X_copy, y_copy, test_size=0.2, random_state=42)

scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)
X_train_copy = scale.fit_transform(X_train_copy)
X_test_copy = scale.transform(X_test_copy)
                                                                       
lr = LogisticRegression()
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs', 'saga'],
    'max_iter': [100, 1000, 10000]
}

grid_base_line_lr = GridSearchCV(estimator=lr, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1, error_score='raise')
grid_base_line_lr.fit(X_train, y_train)
y_pred_lr = grid_base_line_lr.predict(X_test)
print(classification_report(y_test, y_pred_lr))

grid_base_line_lr.fit(X_train_copy, y_train_copy)
y_pred_lr_copy = grid_base_line_lr.predict(X_test_copy)
print(classification_report(y_test_copy, y_pred_lr_copy))


# In[26]:


df


# In[27]:


df_copy


# # 5 Checking and dealing outliers using boxplot and zscore 
# 
# I forgot to check for outliers
# So removing outliers didn't improve the accuracy and the performance in general. I think i need to use different models 
# 

# In[28]:


df


# In[29]:


fig, axs = plt.subplots(7, 2, figsize=(12, 24))
indices = 0
axs = axs.flatten()
for i in df_copy.columns:
    sns.boxplot(df[i], ax=axs[indices])
    axs[indices].set_xlabel(i)  # Set x-label
    indices += 1
plt.tight_layout()  # Adjust the layout
plt.show()


# In[30]:


def remove_outliers(df, columns):
    for column in columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5*iqr
        upper = q3 + 1.5*iqr
        df = df[(df[column] >= lower) & (df[column] <= upper )]
    return df

columns = ['ph', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
df_cleaned = remove_outliers(df, columns)


# In[31]:


df_cleaned


# In[32]:


param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs', 'saga'],
    'max_iter': [100, 1000, 10000]
}
X, y = df_cleaned.drop('Potability', axis=1), df_cleaned['Potability']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)
lr = LogisticRegression()
grid_remove_outliers = GridSearchCV(estimator=lr, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)
grid_remove_outliers.fit(X_train, y_train)
y_pred = grid_remove_outliers.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[33]:


df


# In[34]:


xg = XGBClassifier()
param_grid_xg = {
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 500],
    'max_depth': [3, 5, 7],
    'gamma': [0.0, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
}
grid_xg = GridSearchCV(estimator=xg, param_grid=param_grid_xg, cv=5, verbose=2, n_jobs=-1)
grid_xg.fit(X_train, y_train)
y_pred_xg = grid_xg.predict(X_test)
accuracy_xg = accuracy_score(y_test, y_pred_xg)
print(accuracy_xg)
print(classification_report(y_test, y_pred_xg))


# - So the xgboost classifier model i just used is applied for the dataframe after removing outliers but with imbalanced target. Down here will be xgboost after removing outliers and applying on to balanced taret data
# 

# In[35]:


df.Potability.value_counts()


# In[36]:


df_remove_balance = pd.concat([X_resampled, y_resampled], axis=1)


# In[37]:


df_remove_balance


# In[38]:


df_remove_balance.isna().sum()


# In[39]:


X = df_remove_balance.drop('Potability', axis=1)
imputer = KNNImputer()
X_imputed = imputer.fit_transform(X)
df_ = pd.DataFrame(X_imputed, columns=X.columns)


# In[40]:


df_


# In[41]:


df_remove_balance = pd.concat([df_, y_resampled], axis=1)


# In[42]:


df_remove_balance


# In[43]:


X, y = df_remove_balance.drop('Potability', axis=1), df_remove_balance['Potability']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
xg = XGBClassifier()
param_grid_xg = {
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 500],
    'max_depth': [3, 5, 7],
    'gamma': [0.0, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
}
grid_xg = GridSearchCV(estimator=xg, param_grid=param_grid_xg, cv=5, verbose=2, n_jobs=-1)
grid_xg.fit(X_train, y_train)
y_pred_xg = grid_xg.predict(X_test)
accuracy_xg = accuracy_score(y_test, y_pred_xg)
print(accuracy_xg)
print(classification_report(y_test, y_pred_xg))


# In[44]:


X, y = df_remove_balance.drop('Potability', axis=1), df_remove_balance['Potability']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)
xg = XGBClassifier()
param_grid_xg = {
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 500],
    'max_depth': [3, 5, 7],
    'gamma': [0.0, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
}
grid_xg = GridSearchCV(estimator=xg, param_grid=param_grid_xg, cv=5, verbose=2, n_jobs=-1)
grid_xg.fit(X_train, y_train)
y_pred_xg = grid_xg.predict(X_test)
accuracy_xg = accuracy_score(y_test, y_pred_xg)
print(accuracy_xg)
print(classification_report(y_test, y_pred_xg))


# # I have removed outliers, balanced the target and using XGB to check if those methods are optimal. After that i will use KBest and PCA then check for other models and do hyperparameters tuning 
# 
# # One mistake that i made is that i didn't scale the features, but the accuracy increase as well as the recall in the class 1 to 0.69 and 0.65 respectively
# 
# # And after removing outliers and balancing the target, the accuracy incrased to 70%. I will try to do KBest and PCA with other models now 
# 
# 

# In[45]:


df_remove_balance


# In[46]:


df_remove_balance.Potability.value_counts()


# In[47]:


X, y = df_remove_balance.drop('Potability', axis=1), df_remove_balance['Potability']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

k = 10


# In[48]:


selector = SelectKBest(score_func=f_classif, k=k)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

selected_features_indices = selector.get_support(indices=True)

selected_features = X.columns[selected_features_indices]
print('Selected features: ', selected_features)


# In[49]:


xg = XGBClassifier()
param_grid_xg = {
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 500],
    'max_depth': [3, 5, 7],
    'gamma': [0.0, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
}
grid_xg = GridSearchCV(estimator=xg, param_grid=param_grid_xg, cv=5, verbose=2, n_jobs=-1)
grid_xg.fit(X_train_selected, y_train)
y_pred_kbest = grid_xg.predict(X_test_selected)
accuracy_xg_kbest = accuracy_score(y_test, y_pred_kbest)
print(accuracy_xg_kbest)
print(classification_report(y_test, y_pred_kbest))


# # So i did use KBest, first i tried k=5 which droped the score down to 63 % which is shitty, then i increased the k to 10 and the score increase as well to 69% which is not notable nor good compared to previous score
# 
# # I will try other models: Random Forest 

# In[50]:


rfc = RandomForestClassifier()
param_grid_rfc = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

grid_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid_rfc, cv=5, verbose=2, n_jobs=-1)
grid_rfc.fit(X_train_selected, y_train)
y_pred_kbest = grid_rfc.predict(X_test_selected)
accuracy_rfc_kbest = accuracy_score(y_test, y_pred_kbest)
print(accuracy_rfc_kbest)
print(classification_report(y_test, y_pred_kbest))


# In[51]:


X, y = df_remove_balance.drop('Potability', axis=1), df_remove_balance['Potability']
X_train, X_test, y_train, y_test =train_test_split(X,y, test_size=0.2, random_state=42)
scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)
rfc = RandomForestClassifier()
param_grid_rfc = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

grid_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid_rfc, cv=5, verbose=2, n_jobs=-1)
grid_rfc.fit(X_train, y_train)
y_pred = grid_rfc.predict(X_test)
accuracy_rfc = accuracy_score(y_test, y_pred)
print(accuracy_rfc)
print(classification_report(y_test, y_pred))


# # So i used random forest without Kbest, removed outliers, balanced the target, and the final for today is 72% 
# 

# In[53]:


df_remove_balance


# In[55]:


df_remove_balance.skew()


# In[58]:


for col in df_remove_balance.columns:
    plt.figure(figsize=(12, 8))
    sns.histplot(df_remove_balance[col])


# In[60]:


df_remove_balance.tds_level.value_counts()


# In[62]:


df_remove_balance.drop('tds_level', axis=1, inplace=True)


# In[63]:


X, y = df_remove_balance.drop('Potability', axis=1), df_remove_balance['Potability']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test  = scale.transform(X_test)
rfc = RandomForestClassifier(random_state=42)
param_grid_rfc = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

grid_search_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid_rfc, verbose=2, n_jobs=-1)
grid_search_rfc.fit(X_train, y_train)
y_pred = grid_search_rfc.predict(X_test)
accur = accuracy_score(y_test, y_pred)
print('Accuracy score: ', accur)
print(classification_report(y_test, y_pred))


# In[84]:


from tensorflow.keras.layers import Dropout
X = df_cleaned.drop('Potability', axis=1)
y = df_cleaned['Potability']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1], )),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# In[83]:


df_cleaned


# # I would like to use bagging to check for improvement 
# 

# In[ ]:




