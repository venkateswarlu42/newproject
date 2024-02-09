#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install lightbgm as lgb


# In[5]:


import numpy as np
import pandas as pd
import pyforest
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import warnings
warnings.filterwarnings("ignore")


# In[7]:


USE_DATA_LEAK='Y' ## Set to Y to exploit data leak

RAND_VAL=42
num_folds=5 
n_est=3500 


# In[9]:


df_train = pd.read_csv('Churn_Modelling.csv')
print(df_train.columns)
df_train.head(10)


# In[10]:


df_test = pd.read_csv('Churn_Modelling.csv')
df_test_ov = df_test.copy()
df_test.head()


# In[11]:


scale_cols = ['Age','CreditScore', 'Balance','EstimatedSalary']

for c in scale_cols:
    min_value = df_train[c].min()
    max_value = df_train[c].max()
    df_train[c+"_scaled"] = (df_train[c] - min_value) / (max_value - min_value)
    df_test[c+"_scaled"] = (df_test[c] - min_value) / (max_value - min_value)


# In[12]:


def getFeats(df):
    
    df['IsSenior'] = df['Age'].apply(lambda x: 1 if x >= 60 else 0)
    df['IsActive_by_CreditCard'] = df['HasCrCard'] * df['IsActiveMember']
    df['Products_Per_Tenure'] =  df['Tenure'] / df['NumOfProducts']
    df['AgeCat'] = np.round(df.Age/20).astype('int').astype('category')
    df['Sur_Geo_Gend_Sal'] = df['Surname']+df['Geography']+df['Gender']+np.round(df.EstimatedSalary).astype('str')
    
    return df


# In[14]:


df_train = getFeats(df_train)
df_test = getFeats(df_test)

feat_cols=df_train.columns.drop(['RowNumber','Exited'])
feat_cols=feat_cols.drop(scale_cols)
print(feat_cols)
df_train.head()


# In[15]:


X=df_train[feat_cols]
y=df_train['Exited']
##
cat_features = np.where(X.dtypes != np.float64)[0]
cat_features


# In[26]:


"Mean AUC: ",np.mean(auc_vals)


# In[35]:


df_train.hist(column='Exited', bins=20, range=[0,1],figsize=(12,6))
plt.show()


# In[38]:


df_test.hist(column='Exited', bins=10, range=[0,1],figsize=(16,8))
plt.show()


# In[40]:


df_train['Gender'] = df_train['Gender'].map({'Male': 1, 'Female': 0})


# In[42]:


plt.figure(figsize=(15, 6))

sns.histplot(data=df_train, x='Age', hue='Exited',  multiple="stack",kde=True, palette="viridis")
plt.title('Age Distribution by Exited Status')

plt.tight_layout()
plt.show()


# In[44]:


mode = df_train['Age'][df_train['Exited'] == 0].mode()[0]
mean = df_train['Age'][df_train['Exited'] == 0].mean()
median = df_train['Age'][df_train['Exited'] == 0].median()
mode_exit = df_train['Age'][df_train['Exited'] == 1].mode()[0]
mean_exit= df_train['Age'][df_train['Exited'] == 1].mean()
median_exit = df_train['Age'][df_train['Exited'] == 1].median()

print("-----------------------------------------------------")
print("|    Statistics    |  Exited = 0 |  Exited= 1       |")
print("-----------------------------------------------------")
print(f"| Mode             |  {mode:<9}  |  {mode_exit:<14}  |")
print(f"| Median           |  {median:<9}  |  {median_exit:<14}  |")
print(f"| Mean             |  {mean:<9.2f}  |  {mean_exit:<14.2f}  |")
print("-----------------------------------------------------")


# In[45]:


q1 = df_test['Age'].quantile(0.25)
q3 = df_test['Age'].quantile(0.75)

q1_exit_0 = df_test['Age'][df_test['Exited'] == 0].quantile(0.25)
q3_exit_0 = df_test['Age'][df_test['Exited'] == 0].quantile(0.75)

q1_exit_1 = df_test['Age'][df_test['Exited'] == 1].quantile(0.25)
q3_exit_1 = df_test['Age'][df_test['Exited'] == 1].quantile(0.75)


print("Quartiles for Age Distribution when Exited = 0:")
print(f"Q1: {q1_exit_0}, Q3: {q3_exit_0}\n")

print("Quartiles for Age Distribution when Exited = 1:")
print(f"Q1: {q1_exit_1}, Q3: {q3_exit_1}")


# In[47]:


df_train = df_train[df_train['Exited'] == 1]

plt.figure(figsize=(15, 6))

sns.histplot(data=df_train, x='Age', hue='HasCrCard', multiple="stack", kde=True, palette="viridis")

plt.title('Age Distribution by Credit Card Ownership for Exited Individuals')
plt.xlabel('Age')
plt.ylabel('Count')

plt.tight_layout()
plt.show()


# In[48]:


plt.figure(figsize=(15, 6))

sns.histplot(data=df_train, x='Gender', hue='Exited',  multiple="stack", palette="viridis")
plt.title('Gender Distribution by Exited Status')

plt.tight_layout()
plt.show()


# In[50]:


plt.figure(figsize=(15,5))

plt.subplot(1, 2, 1)
sns.histplot(data=df_test.loc[df_test['Exited'] == 0], x="EstimatedSalary", kde=True, color=sns.color_palette("viridis")[1])
plt.title("Estimated Salary Distribution for Not Exited")

plt.subplot(1, 2, 2)
sns.histplot(data=df_train.loc[df_train['Exited'] == 1], x="EstimatedSalary", kde=True, color=sns.color_palette("viridis")[1])
plt.title("Estimated Salary Distribution for Exited")

plt.tight_layout()
plt.show()


# In[51]:


plt.figure(figsize=(15, 6))

sns.histplot(data=df_train, x='IsActiveMember', hue='Exited',  multiple="stack",kde=True,  palette="viridis")
plt.title('Is Active Member Distribution by Exited Status')

plt.tight_layout()
plt.show()


# In[52]:


plt.figure(figsize=(15, 6))

sns.histplot(data=df_train, x='NumOfProducts', hue='Exited',  multiple="stack",kde=True,  palette="viridis")
plt.title('Num Of Products Distribution by Exited Status')

plt.tight_layout()
plt.show()


# In[57]:


churn_count = df_test[df_test['Exited'] == 1].groupby('Geography').size().reset_index(name='churn_count')

non_churn_count = df_test[df_test['Exited'] == 0].groupby('Geography').size().reset_index(name='non_churn_count')

combined_count = churn_count.merge(non_churn_count, on='Geography')

total_count = df_test['Geography'].value_counts().reset_index()
total_count.columns = ['Geography', 'total_count']
combined_count = combined_count.merge(total_count, on='Geography')
combined_count['churn_percentage'] = (combined_count['churn_count'] / combined_count['total_count']).round(4) * 100
combined_count


# In[59]:


df_test['Geography'] = df_test['Geography'].map({'France': 1, 'Germany': 2, 'Spain': 3})


# In[60]:


df_test.drop(['CustomerId', 'Surname'], axis=1, inplace=True)


# In[61]:


df_test.head()


# In[62]:


X_train, X_test, y_train, y_test =train_test_split(df_test.drop('Exited',axis=1),df_test['Exited'],test_size=0.3)


# In[72]:


from sklearn.tree import DecisionTreeClassifier

modelo_tree = DecisionTreeClassifier(max_depth=5)




# In[73]:


print(modelo_tree)


# In[ ]:




