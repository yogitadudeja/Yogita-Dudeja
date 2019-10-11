
# coding: utf-8

# In[26]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split

import seaborn as sns


# In[2]:


train= pd.read_csv(r"C:\Users\Ravinder Kumar\Desktop\train.csv")
test= pd.read_csv(r"C:\Users\Ravinder Kumar\Desktop\test.csv")
train.head()


# In[3]:


train.info()


# # Plotting on Barchart

# In[4]:


class_pivot = train.pivot_table(index="Pclass",values="Survived")
class_pivot.plot.bar()
plt.show()


# In[5]:


sex_pivot = train.pivot_table(index="Sex",values="Survived")
sex_pivot.plot.bar()
plt.show()


# # Plotting Age on Histogram (because it is continuos variable)
survived = train[train["Survived"] == 1]
died = train[train["Survived"] == 0]
survived["Age"].plot.hist(alpha=0.5,color='red',bins=50)
died["Age"].plot.hist(alpha=0.5,color='blue',bins=50)
plt.legend(['Survived','Died'])
plt.show()
# # Mapping sex into numerical value:{0,1}

# In[16]:


sex_mapping = {"male": 0, "female": 1}
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)

train.head()


# # Removing null values

# In[17]:


train['Age'] = train['Age'].fillna(0)
test['Age'] = test['Age'].fillna(0)


# In[45]:


predictors = train.drop(['Survived', 'PassengerId','Name','SibSp','Parch', 'Ticket','Fare','Embarked'],axis=1)
target = train["Survived"]


# 
# # Training model

# As we have to classify our output into survived, died so we have to use classifier. Naive baeys is the classifier which can be used on even smaller datasets. So we use it here

# In[41]:


x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.20, random_state = 0)


# In[49]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gaussian)


# Now we use Support Vector Machine Model 
# 

# In[48]:


from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_svc)

