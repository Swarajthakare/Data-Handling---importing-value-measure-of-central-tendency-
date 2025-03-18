#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ### load data set

# In[2]:


df = pd.read_csv(r'B:\Demo Datasets\House price prediction advance regration technique\train.csv')
df.shape


# In[3]:


pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)


# ### data view

# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.describe()


# In[7]:


df.info()


# In[8]:


df.isna().sum()


# ### droping missing values columns greater then 20

# In[9]:


nan_var_per = df.isna().sum()/df.shape[0]*100
nan_var_per


# In[10]:


miss_value_gre20 = nan_var_per[nan_var_per>20].keys()
miss_value_gre20


# In[11]:


df2_drop_clm = df.drop(columns = miss_value_gre20)
df2_drop_clm.shape


# ### filling\cleaning int value variable and showing destribution

# In[12]:


df3_num_var = df2_drop_clm.select_dtypes(include = ['int64','float64'])
df3_num_var


# In[13]:


plt.figure(figsize = (10,10))
sns.heatmap(df3_num_var.isna())


# In[14]:


df3_num_var[df3_num_var.isna().any(axis = 1)]


# In[15]:


df3_num_var.isna().sum()


# In[16]:


missing_num_var = [var for var in df3_num_var.columns if df3_num_var[var].isnull().sum()>0]
missing_num_var


# In[17]:


plt.figure(figsize = (10,10))
sns.set()
for i,variable in enumerate(missing_num_var):
    plt.subplot(2,2,i+1)
    sns.distplot(df3_num_var[variable],bins = 20, kde_kws = {'linewidth': 5,'color':'#DC143C'})


# ### NAN VALUES FIILING BY MEAN

# In[18]:


df4_num_mean = df3_num_var.fillna(df3_num_var.mean())
df4_num_mean.isna().sum().sum()


# In[19]:


plt.figure(figsize = (10,10))
sns.set()
for i,variable in enumerate(missing_num_var):
    plt.subplot(2,2,i+1)
    sns.distplot(df3_num_var[variable],bins = 20, kde_kws = {'linewidth': 5,'color':'#DC143C'},label = 'orignal')
    sns.distplot(df4_num_mean[variable],bins = 20, kde_kws = {'linewidth': 4,'color':'g'},label = 'mean')
    plt.legend()


# ### NAN VALUES FIILING BY median 

# In[20]:


df5_num_median = df3_num_var.fillna(df3_num_var.median())
df5_num_median.isna().sum().sum()


# In[21]:


plt.figure(figsize = (10,10))
sns.set()
for i,variable in enumerate(missing_num_var):
    plt.subplot(2,2,i+1)
    sns.distplot(df3_num_var[variable],bins = 20, kde_kws = {'linewidth': 5,'color':'#DC143C'},label = 'orignal')
    sns.distplot(df4_num_mean[variable],bins = 20, kde_kws = {'linewidth': 4,'color':'g'},label = 'mean')
    sns.distplot(df5_num_median[variable],bins = 20,kde_kws = {'linewidth':3,'color':'k'},label = 'median')
    plt.legend()


# In[22]:


for i,char in enumerate(missing_num_var):
    plt.figure(figsize=(10,10))
    plt.subplot(3,1,1)
    sns.boxplot(df[variable])
    plt.subplot(3,1,2)
    sns.boxplot(df4_num_mean[variable])
    plt.subplot(3,1,3)
    sns.boxplot(df5_num_median[variable])


# In[ ]:





# In[ ]:




