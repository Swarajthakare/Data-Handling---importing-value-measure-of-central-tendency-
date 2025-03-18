#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv(r'B:\Demo Datasets\House price prediction advance regration technique\train.csv')
df.shape


# In[3]:


pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)


# In[4]:


df.head()


# In[5]:


null = df.isnull().sum()/df.shape[0]*100
null


# In[6]:


missing_value_gre20 = null[null>20].keys()
missing_value_gre20


# In[7]:


df2_drop_clm = df.drop(columns = missing_value_gre20)
df2_drop_clm.shape


# In[8]:


df3_num = df2_drop_clm.select_dtypes(include = ['int64','float64'])
df3_num.shape


# In[9]:


df3_num.isnull().sum()


# In[10]:


num_var_missing = [var for var in df3_num.columns if df3_num[var].isnull().sum()>0]
num_var_missing


# In[11]:


df3_num[num_var_missing][df3_num[num_var_missing].isnull().any(axis = 1)]


# In[12]:


df['LotConfig'].unique()


# In[13]:


df[df.loc[:,'LotConfig'] == 'Inside']


# In[14]:


df[df.loc[:,'LotConfig'] == 'Inside'].shape


# In[15]:


df[df.loc[:,'LotConfig'] == 'Inside']['LotFrontage']


# In[16]:


df[df.loc[:,'LotConfig'] == 'Inside']['LotFrontage'].replace(np.nan,df[df.loc[:,'LotConfig'] == 'Inside']['LotFrontage'].mean())


# In[17]:


df_copy = df.copy()

for var_class in df['LotConfig'].unique():
    df_copy.update(df[df.loc[:,'LotConfig'] == var_class ]['LotFrontage'].replace(np.nan,df[df.loc[:,'LotConfig'] == var_class]['LotFrontage'].mean()))


# In[18]:


df_copy.isnull().sum()


# In[19]:


df_copy = df.copy()
num_vars_miss = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']
cat_vars = ['LotConfig','MasVnrType','GarageType']
for cat_var,num_var_miss in zip(cat_vars,num_vars_miss):
    for var_class in df[cat_var].unique():
        df_copy.update(df[df.loc[:,cat_var] == var_class ]['LotFrontage'].replace(np.nan,df[df.loc[:,cat_var] == var_class][num_var_miss].mean()))


# In[20]:


df_copy[num_vars_miss].isnull().sum()


# In[21]:


df_copy[df_copy[['MasVnrArea']].isnull().any(axis = 1)]


# In[22]:


df_copy[df_copy[['MasVnrArea']].isnull().any(axis = 1)].shape


# In[23]:


df_copy[df_copy[['GarageYrBlt']].isnull().any(axis = 1)]


# In[24]:


df_copy = df.copy()
num_vars_miss = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']
cat_vars = ['LotConfig','Exterior2nd','KitchenQual']
for cat_var,num_var_miss in zip(cat_vars,num_vars_miss):
    for var_class in df[cat_var].unique():
        df_copy.update(df[df.loc[:,cat_var] == var_class ][num_var_miss].replace(np.nan,df[df.loc[:,cat_var] == var_class][num_var_miss].mean()))


# In[25]:


df_copy[num_vars_miss].isnull().sum()


# In[26]:


plt.figure(figsize = (10,10))
sns.set()
for i,var in enumerate(num_vars_miss):
    plt.subplot(2,2,i+1)
    sns.distplot(df[var], bins = 20, kde_kws = {'linewidth':8, 'color':'red'},label = 'orignal')
    sns.distplot(df_copy[var],bins = 20, kde_kws = {'linewidth':5,'color':'green'},label = 'clean')
    plt.legend()
    


# In[27]:


df_copy_median = df.copy()
num_vars_miss = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']
cat_vars = ['LotConfig','Exterior2nd','KitchenQual']
for cat_var,num_var_miss in zip(cat_vars,num_vars_miss):
    for var_class in df[cat_var].unique():
        df_copy_median.update(df[df.loc[:,cat_var] == var_class ][num_var_miss].replace(np.nan,df[df.loc[:,cat_var] == var_class][num_var_miss].median()))


# In[28]:


df_copy_median[num_vars_miss].isnull().sum()


# In[29]:


plt.figure(figsize = (10,10))
sns.set()
for i,var in enumerate(num_vars_miss):
    plt.subplot(2,2,i+1)
    sns.distplot(df[var], bins = 20, kde_kws = {'linewidth':8, 'color':'red'},label = 'orignal')
    sns.distplot(df_copy[var],bins = 20, kde_kws = {'linewidth':5,'color':'green'},label = 'mean')
    sns.distplot(df_copy_median[var],bins = 20,kde_kws = {'linewidth':4,'color':'k'},label = 'median')
    plt.legend()


# In[30]:


for i,var in enumerate(num_vars_miss):
    plt.figure(figsize = (10,10))
    plt.subplot(3,1,1)
    sns.boxplot(df[var])
    plt.subplot(3,1,2)
    sns.boxplot(df_copy[var])
    plt.subplot(3,1,3)
    sns.boxplot(df_copy_median[var])
    


# In[ ]:




