#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ### load dataset

# In[2]:


df = pd.read_csv(r'B:\Demo Datasets\House price prediction advance regration technique\train.csv')
df.shape


# In[3]:


pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


cat_vars = df.select_dtypes(include = 'object')
cat_vars.shape


# In[7]:


cat_vars.head()


# In[8]:


miss_value_per = cat_vars.isna().mean()*100
miss_value_per


# In[9]:


drop_vars = miss_value_per[miss_value_per>20].keys()
drop_vars


# In[10]:


cat_vars.drop(columns = drop_vars,axis = 1,inplace = True)
cat_vars.shape


# In[11]:


cat_vars.isna().sum()


# In[12]:


isnull_per = cat_vars.isna().mean()*100
miss_vars = isnull_per[isnull_per>0].keys()
miss_vars


# In[13]:


#filling manualy
cat_vars['MasVnrType'].fillna('Missing')


# In[14]:


cat_vars['MasVnrType'].mode()


# In[15]:


cat_vars['MasVnrType'].value_counts()


# In[16]:


cat_vars['MasVnrType'].fillna(cat_vars['MasVnrType'].mode()[0])


# In[17]:


for var in miss_vars:
    cat_vars[var].fillna(cat_vars[var].mode()[0],inplace = True)
    print(var,'=',cat_vars[var].mode()[0])


# In[18]:


cat_vars.isna().sum()


# In[19]:


plt.figure(figsize=(16,9))
for i,var in enumerate(miss_vars):
    plt.subplot(4,3,i+1)
    plt.hist(cat_vars[var],label='Imput')
    plt.hist(df[var].dropna(),label = 'original')
    plt.legend()


# In[20]:


df.update(cat_vars)
df.drop(columns = drop_vars ,inplace = True)


# In[21]:


df.select_dtypes(include = 'object').isna().sum().sum()


# In[ ]:




