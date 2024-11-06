#!/usr/bin/env python
# coding: utf-8

# # Zomato data analysis

# import libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv('Zomato data .csv')
df


# what type of resturant do the majority of customers order from?

# In[3]:


def handleRate(value):
    value=str(value).split('/')
    value=value[0];
    return float(value)
df['rate']=df['rate'].apply(handleRate)
df


# In[4]:


df.head()


# In[5]:


df.info


# In[6]:


df.describe()


# In[7]:


df.isnull().sum()


# In[9]:


sns.countplot(x=df['listed_in(type)'])
plt.xlabel('type of resturanr')
plt.legend()


# # conclusion of first question dining category is majority types resturant

# how many votes has each type of resturant received from customer

# In[14]:


group_data=df.groupby('listed_in(type)')['votes'].sum()
result=pd.DataFrame({'votes':group_data})


# In[21]:


plt.plot(result,c='blue',marker='*')
plt.xlabel('types of resturant',c='black',size=15)
plt.ylabel('votes',c='black',size=15)
plt.show()


# # conclusion :- dining resturant has recieved maximum votes

# # what are the ratings that the majority of resturants have received?

# In[22]:


df.head()


# In[23]:


plt.hist(df['rate'],bins=5)
plt.title('rating distribution')
plt.show()


# conclusion :- the majority resturants received rating from 3.5 to 4

# # zomato has observed most couple ordre most of their food online. what is their averge spending on each order ?

# In[24]:


df.head()


# In[25]:


couple_data=df['approx_cost(for two people)']
sns.countplot(x=couple_data)


# conclusion :- the majority of couples prefer resturant with an approximate cost of 300 rupees

# # which mode (online or offline) has received the maximum rating?

# In[26]:


df.head()


# In[30]:


plt.figure(figsize=(6,6))
sns.boxplot(x='online_order',y='rate',data=df)
plt.show()


# conclusion :- offline ordered received lower rating in comaprision to online order

# # which type of resturant received more offline orders, so that zomato can prefer customer with some good offers

# In[31]:


df.head()


# In[34]:


pivot_table=df.pivot_table(index='listed_in(type)',columns='online_order',aggfunc='size',fill_value=0)
sns.heatmap(pivot_table,annot=True,cmap='Paired',fmt='d')
plt.title('HeatMap')
plt.xlabel('online_order')
plt.ylabel('listed_in(type)')
plt.show()


# In[ ]:




