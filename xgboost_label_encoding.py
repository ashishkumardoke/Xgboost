
# coding: utf-8

# In[28]:


# binary classification, breast cancer dataset, label and one hot encoded
from numpy import column_stack
from pandas import read_csv
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


import numpy as np 
import pandas as pd 
# load data
dataset= read_csv('cancer.csv')

dataset.head()


# In[14]:


del dataset['Unnamed: 32']


# In[15]:


del dataset['id']


# In[16]:


covariates = list(dataset.columns.values)
cov = list(covariates)
cov.remove('diagnosis')


# In[17]:


X= dataset[cov]
Y= dataset['diagnosis']


# In[21]:


label_encoder= LabelEncoder()
label_encoder=label_encoder.fit(Y)
label_encoded_y=label_encoder.transform(Y)


# In[25]:


# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, label_encoded_y , test_size=test_size, random_state=seed)


# In[26]:


# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)


# In[27]:


accuracy_score(y_test,y_pred)


# In[ ]:




