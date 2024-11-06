#!/usr/bin/env python
# coding: utf-8

# ### import Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


df= pd.read_csv(r"C:\Users\91805\Downloads\archive (1)\data.csv")
df.head()


# In[4]:


df.tail()


# In[5]:


df.shape


# In[6]:


df.describe().T


# In[7]:


df.diagnosis.unique()


# In[8]:


df['diagnosis'].value_counts()


# In[10]:


sns.pairplot(df,vars=['radius_mean','texture_mean','perimeter_mean'],hue='diagnosis')


# In[11]:


df.drop('id',axis=1,inplace=True)
df.drop('Unnamed: 32',axis=1,inplace=True)


# In[12]:


df.head()


# In[13]:


df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})
df.head()


# In[14]:


df.isnull().sum()


# In[15]:


df.corr()


# In[16]:


plt.hist(df['diagnosis'], color='g')
plt.title('Plot_Diagnosis (M=1 , B=0)')
plt.show()


# In[17]:


plt.figure(figsize=(20,20))
sns.heatmap(df.corr(), annot=True)


# In[18]:


cols = ['diagnosis',
        'radius_mean', 
        'texture_mean', 
        'perimeter_mean', 
        'area_mean', 
        'smoothness_mean', 
        'compactness_mean', 
        'concavity_mean',
        'concave points_mean', 
        'symmetry_mean', 
        'fractal_dimension_mean']
sns.pairplot(data=df[cols], hue='diagnosis', palette='rocket')


# In[20]:


cols = ['radius_worst', 
        'texture_worst', 
        'perimeter_worst', 
        'area_worst', 
        'smoothness_worst', 
        'compactness_worst', 
        'concavity_worst',
        'concave points_worst', 
        'symmetry_worst', 
        'fractal_dimension_worst']
df = df.drop(cols, axis=1)

# then, drop all columns related to the "perimeter" and "area" attributes
cols = ['perimeter_mean',
        'perimeter_se', 
        'area_mean', 
        'area_se']
df = df.drop(cols, axis=1)

# lastly, drop all columns related to the "concavity" and "concave points" attributes
cols = ['concavity_mean',
        'concavity_se', 
        'concave points_mean', 
        'concave points_se']
df = df.drop(cols, axis=1)

# verify remaining columns
df.columns


# ### Building Model

# In[22]:


##Building Model
X=df.drop(['diagnosis'],axis=1)
y = df['diagnosis']


# In[23]:


from sklearn.model_selection import train_test_split


# In[24]:


X_train, X_test,y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=40)


# ### Feature Scaling

# In[25]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()

X_train=ss.fit_transform(X_train)
X_test=ss.fit_transform(X_test)


# ### Logistics Regression

# In[26]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()


# In[27]:


model1=lr.fit(X_train,y_train)
prediction1=model1.predict(X_test)


# ### Confusion_Matrix

# In[28]:


from sklearn.metrics import confusion_matrix


# In[29]:


cm=confusion_matrix(y_test,prediction1)
cm


# In[30]:


sns.heatmap(cm,annot=True)
plt.savefig('h.png')


# In[31]:


TP=cm[0][0]
TN=cm[1][1]
FN=cm[1][0]
FP=cm[0][1]
print('Testing Accuracy:',(TP+TN)/(TP+TN+FN+FP))


# In[32]:


from sklearn.metrics import accuracy_score


# In[33]:


accuracy_score(y_test,prediction1)


# ### Decision Tree

# In[34]:


#Decision Tree
from sklearn.tree import DecisionTreeClassifier


# In[35]:


dtc=DecisionTreeClassifier()
model2=dtc.fit(X_train,y_train)
prediction2=model2.predict(X_test)
cm2= confusion_matrix(y_test,prediction2)


# In[36]:


cm2


# In[37]:


accuracy_score(y_test,prediction2)


# ### Random Forest Classifier

# In[38]:


from sklearn.ensemble import RandomForestClassifier


# In[39]:


rfc=RandomForestClassifier()
model3 = rfc.fit(X_train, y_train)
prediction3 = model3.predict(X_test)
confusion_matrix(y_test, prediction3)


# In[40]:


accuracy_score(y_test, prediction3)


# In[41]:


from sklearn.metrics import classification_report
print(classification_report(y_test, prediction3))


# In[42]:


print(classification_report(y_test, prediction1))

print(classification_report(y_test, prediction2))


# ## KNN

# In[43]:


##KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


# In[44]:


models=[]

models.append(('KNN', KNeighborsClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))


# ## Cross-Validation

# In[45]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[49]:


SVM = SVC()
SVM.fit(X_train, y_train)
predictions= SVM.predict(X_test)
print(accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))


# In[ ]:




