
# coding: utf-8

# In[50]:


from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot
from sklearn.feature_selection import SelectFromModel


import numpy as np 
import pandas as pd 


# In[2]:


dataset = pd.read_csv("data.csv")


# In[3]:


del dataset['Unnamed: 32']


# In[4]:


dataset.head()


# In[5]:


del dataset['id']


# In[6]:


dataset.head()


# In[7]:


covariates = list(dataset.columns.values)
cov = list(covariates)
cov.remove('diagnosis')


# X= dataset[cov]
# Y= dataset['diagnosis']

# In[ ]:



# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)


# In[27]:


# fit model no training data
model = XGBClassifier()

kfold = StratifiedKFold(n_splits=7, random_state=7)

results = cross_val_score(model, X, Y, cv=kfold)

print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

#model.fit(X_train, y_train)

# make predictions for test data

#y_pred = model.predict(X_test)


# max_depth = range(1, 11, 2)
# print(max_depth)
# param_grid = dict(max_depth=max_depth)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
# grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold, verbose=1)
# grid_result = grid_search.fit(X, label_encoded_y)
# 
# # summarize results
# 
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
# 	print("%f (%f) with: %r" % (mean, stdev, param))
#     
# # plot
# pyplot.errorbar(max_depth, means, yerr=stds)
# pyplot.title("XGBoost max_depth vs Log Loss")
# pyplot.xlabel('max_depth')
# pyplot.ylabel('Log Loss')
# pyplot.savefig('max_depth.png')

# In[39]:


from matplotlib import pyplot
model.fit(X,Y)

print(model.feature_importances_)
pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)


#plot_tree(model, num_trees=4)
pyplot.show()


# In[47]:


from xgboost import plot_importance
plot_importance(model)
pyplot.show()
from numpy import sort


# In[51]:


model.fit(X_train, y_train)
y_pred= model.predict(X_test)

accuracy=accuracy_score(y_test, y_pred)

thresholds= sort(model.feature_importances_)
for thresh in thresholds:
    
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train)
	# train model
    selection_model = XGBClassifier()
    selection_model.fit(select_X_train, y_train)
	# eval model
    select_X_test = selection.transform(X_test)
    y_pred = selection_model.predict(select_X_test)


# In[70]:


eval_set = [(X_test, y_test)]
model.fit(X_train, y_train,eval_metric=["error", "logloss"],eval_set=eval_set,verbose=True)
results = model.evals_result()
#y_pred = model.predict(X_test)
#accuracy = accuracy_score(y_test, y_pred)


# In[52]:


accuracy_score(y_test,y_pred)


# In[71]:


print(results)


# In[72]:


results =model.evals_result()
epochs=len(results['validation_0']['error'])
x_axis=range(0,epochs)


# fig, ax=pyplot.subplots()
# ax.plot(x_axis, results['validation_0']['error'],label='Train')
# ax.plot(x_axis, results['validation_1']['error'],label='Test')
# ax.legend()
# pyplot.ylabel('Log Loss')
# pyplot.title('XGBoost error')
# pyplot.show()
# 

# In[ ]:




