#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# In[2]:


data = pd.read_csv('spam.csv', encoding='latin-1')


# In[4]:


data.head()


# In[5]:


data.info()


# In[6]:


data.isnull().sum()


# In[7]:


data = data[['v1', 'v2']]
data.columns = ['label', 'text']


# In[8]:


print(data.head(5))


# In[9]:


data['label'] = data['label'].apply(lambda x: 1 if x == 'spam' else 0)


# In[10]:


X = data['text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[11]:


tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


# In[12]:


logreg_model = LogisticRegression()
logreg_model.fit(X_train_tfidf, y_train)


# In[13]:


# Test the model
y_train_pred = logreg_model.predict(X_train_tfidf)
y_test_pred = logreg_model.predict(X_test_tfidf)


# In[14]:


train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)


# In[15]:


print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)


# In[16]:


train_report = classification_report(y_train, y_train_pred)
test_report = classification_report(y_test, y_test_pred)


# In[17]:


print("\nTraining Classification Report:\n", train_report)
print("\nTesting Classification Report:\n", test_report)


# In[18]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[19]:


cm_test = confusion_matrix(y_test, y_test_pred)
print("\nConfusion Matrix (Test Set):\n", cm_test)


# In[28]:


# Plot confusion matrix for test set
plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues", xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam'])
plt.title('Confusion Matrix (Test Set)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[21]:


from sklearn.metrics import roc_curve, auc

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, logreg_model.predict_proba(X_test_tfidf)[:, 1])
roc_auc = auc(fpr, tpr)


# In[22]:


plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[35]:


# Plot confusion matrix for test set
plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues", xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam'])
plt.title('Confusion Matrix (Test Set)')
plt.xlabel('y_train')
plt.ylabel('True Label')
plt.show()


# In[39]:


pip install wordcloud


# In[47]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

text = "spam detection".join(data)


wordcloud = WordCloud(background_color='white').generate(text)

plt.figure(figsize=(30,20))
plt.imshow(wordcloud)
plt.axis("off")
plt.title("WordCloud For Text Before StopWords", fontsize=15)
plt.show()


# In[54]:


emails = ["Congratulations! You've won a free vacation. Click the link to claim your prize!",
"Urgent: Your account has been compromised. Click here to secure it now!",
"Make money fast! Join our exclusive program and start earning thousands in just a week!",
"Limited time offer! Buy one, get one free. Click now for incredible deals!",
"You've been selected for a special promotion. Claim your reward by clicking the link below!",
"Act now to receive a 50% discount on all products. Limited stock available!",
"Get rich quick! Invest in our revolutionary scheme and watch your money multiply!",
"Claim your inheritance! You are entitled to a large sum of money. Provide your details to process the transfer.",
"Your computer is infected! Download our antivirus software immediately to protect your data.",
"Win a brand new iPhone X by participating in our survey. Click the link to get started!"]


# In[56]:


emails


# In[ ]:




