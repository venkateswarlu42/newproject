#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 1. to handle the data
import pandas as pd
import numpy as np

# to visualize the dataset
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[2]:


from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler


# In[3]:


from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score


# In[5]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestRegressor


# In[6]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# In[7]:


import warnings
warnings.filterwarnings('ignore')


# In[8]:


df = pd.read_csv('creditcard.csv')


# In[10]:


pd.set_option('display.max_columns', None)


# In[11]:


df.head()


# In[12]:


df.info()


# In[13]:


df.shape


# In[14]:


df.columns


# In[15]:


df.describe()


# In[16]:


from sklearn.preprocessing import Normalizer

normalizer = Normalizer(norm='l2')
print(normalizer.fit_transform(df))


# In[17]:


df['Class'].value_counts()


# In[18]:


legit = df[df['Class'] == 0]
fraud = df[df['Class'] == 1]


# In[19]:


legit.describe()


# In[20]:


legit.Amount.describe()


# In[21]:


fraud.describe()


# In[22]:


fraud.Amount.describe()


# In[23]:


df.groupby('Class').mean()


# In[24]:


legit_sample = legit.sample(n=492)
# Concatenating two DataFrames
new_df = pd.concat([legit_sample, fraud], axis=0)
# Print first 5 rows of the new dataset
new_df.head()


# In[26]:


new_df['Class'].value_counts()


# In[27]:


df.isnull().sum().sort_values(ascending = False)


# In[28]:


new_df.shape


# In[29]:


X = new_df.drop(columns='Class', axis=1)
y = new_df['Class']


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[31]:


print(X.shape, X_train.shape, X_test.shape)


# In[32]:


model = RandomForestClassifier(random_state=42)


# In[33]:


from sklearn.pipeline import Pipeline


# Create a pipeline for each model
pipeline = Pipeline([
    ('model', model)
    ])
    
# Perform cross-validation
scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    
# Calculate mean accuracy
mean_accuracy = scores.mean()
    
# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)
    
# Make predictions on the test data
y_pred = pipeline.predict(X_test)
    
# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
    
print("Model:", RandomForestClassifier())
print("Cross-validation Accuracy:", mean_accuracy)
print("Test Accuracy:", accuracy)
print('Recall Score: ', recall_score(y_test, y_pred))
print('Precision Score: ', precision_score(y_test, y_pred))
print('F1 Score: ', f1_score(y_test, y_pred))

best_model = pipeline
    
# save the best model
import pickle
pickle.dump(best_model, open('iris_model.dot', 'wb'))


# In[34]:


LABELS = ['Normal', 'Fraud'] 
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred) 
plt.figure(figsize =(12, 12)) 
sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, fmt ="d"); 
plt.title("Confusion matrix") 
plt.ylabel('True class') 
plt.xlabel('Predicted class') 
plt.show()


# In[35]:


df_train = new_df.copy()
df_train.head()


# In[36]:


def display_feature_importance(model,percentage ,top_n=34, plot=False):
    # X and y 
    X = df_train.drop('Class',axis=1)
    y = df_train['Class']
    
    #The model is fitted using the features (X) and the target variable (y), and then the feature importances are calculated.
    model.fit(X, y)
    
    # Get feature importance
    feature_importance = model.feature_importances_
    feature_names = X.columns
    
    # Create a DataFrame for better visualization
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    
    # Sort features by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    # A threshold is calculated based on a specified percentage of the top feature importance.
    #  Features with importance above this threshold are selected.
    threshold = percentage / 100 * feature_importance_df.iloc[0]['Importance']
    
    # Select features that meet the threshold
    selected_features = feature_importance_df[feature_importance_df['Importance'] >= threshold]['Feature'].tolist()
    
    #Print Selected Feature 
    print("Selected Features by {} \n \n at threshold {}%; {}".format(model , percentage,selected_features))
    
    if plot==True:
        # Set seaborn color palette to "viridis"
        sns.set(style="whitegrid", palette="viridis")
    
        # Display or plot the top features
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(top_n))
        plt.title('Feature Importance for {}'.format(type(model).__name__))
        plt.show()
        
    # Add 'Exited' to the list of selected features
    selected_features.append('Class')
        
    return selected_features


# In[38]:


imp_fea = ['V14', 'V10', 'V4', 'V7', 'V21', 'V8', 'V20', 'V3', 'V5', 'V11', 'V12', 'V26', 'V17','Class']
df_train = df_train[imp_fea]
df_train.head()


# In[39]:


df_train.shape


# In[40]:


models = ['XGB Classifier', 'RandomForestClassifier']
accuracy_scores = [accuracy, accuracy]

# Find the index of the maximum accuracy
best_accuracy_index = accuracy_scores.index(max(accuracy_scores))

# Print the best model for accuracy
print(f'Best Accuracy: {accuracy_scores[best_accuracy_index]:.2f} with Model: {models[best_accuracy_index]}')


# In[47]:


import numpy as np

# Define the data points
x = np.array(['v4'])
y = np.array(['v7'])


# Print the best-fit line equation
print("Best-fit line)


# In[ ]:




