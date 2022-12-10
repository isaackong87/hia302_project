#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Stage 1 : Irfan data profiling using df_vax_demog_age
#update 7/12/2022


# In[2]:


import pandas as pd
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, date
import os, sys


# In[3]:


#importing vaccination data  using demog-age data and review raw data
df_vax_demog_age =pd.read_csv ("https://raw.githubusercontent.com/mdirfan76/covid19-public/main/vaccination/vax_demog_age.csv")
#fork from github from 9122022


# In[4]:


df1=df_vax_demog_age


# In[5]:


# STEP 1 : REVIEW RAW DATA 
df1


# In[5]:


#STEP 2 : TO REVIEW DIMENSIONS OF OUR DATASET
df1.shape


# In[6]:


# STEP 3 : REVIEW DATA ATTRIBUTES IN OUR DATA
df1.info()
# TO IDENTIFY DATA TYPE, PLAN TO CHANGE DATE TO DATETIME FORMAT


# In[7]:


# STEP 5 SUMMARIZE DATA USING DESCRIPTIVE STATISTICS
df1.describe()


# In[8]:


#to check missing value
df1.isnull().sum()
df1.isna().sum()


# In[9]:


# check duplicate based on all columns
df1.duplicated().sum()


# In[9]:


#Stage 2 : Transform date to datetime series 
#update 7/12/2022


# In[8]:


# to change date to datetime format
df1['date']=pd.to_datetime(df1['date'], yearfirst=True)


# In[12]:


df1.info()
# we identify that date already change to datetime format


# In[5]:


df2 = df1[["date",'state','full_5_11','booster_5_11','full_12_17','booster_12_17','full_60_69','booster_60_69','full_70_79','booster_70_79','full_80','booster_80']]


# In[6]:


df2


# In[15]:


df2.info()


# In[16]:


# to check number of state and its occurance
df2["state"].value_counts(ascending=True)


# In[17]:


df2.shape


# In[9]:


# descriptive data to know the total of vaccination according to category
df2.sum()


# In[19]:


df2.describe()


# In[10]:


# Stage 3  : Using Data Visualizatin
#to check number of people vaccine age 60-69  compare full vaccination and booster

plt.figure(figsize=(15,5))
plt.title ("Number of vaccination age 60-69 overtime", fontsize=15)
plt.xlabel("\nDate",size=15)
plt.ylabel("Numbr of individuals who received daily vaccine \n",size=15)
plt.grid(linewidth =0.2)

plt.plot(df2.date, df2.full_60_69,label="individuals who received full vaccination")
plt.plot(df2.date, df2.booster_60_69, label ="invidual who received 1st booster")
plt.legend(fontsize="15")
plt.show()


# In[28]:


#to check number of people vaccine age 70-79 compare full vaccine and booster

plt.figure(figsize=(15,5))
plt.title ("Number of vaccination age 70-79 overtime", fontsize=15)
plt.xlabel("\nDate",size=15)
plt.ylabel("Numbr of individuals who received daily vaccine \n",size=15)
plt.grid(linewidth =0.2)

plt.plot(df2.date, df2.full_70_79,label="individuals who received full vaccination")
plt.plot(df2.date, df2.booster_70_79, label ="invidual who received 1st booster")
plt.legend(fontsize="15")
plt.show()


# In[29]:


#to check number of people vaccine age >80 compare full vaccine and booster

plt.figure(figsize=(10,5))
plt.title ("Number of vaccination age > 80 overtime", fontsize=15)
plt.xlabel("\nDate",size=15)
plt.ylabel("Numbr of individuals who received daily vaccine \n",size=15)
plt.grid(linewidth =0.2)

plt.plot(df2.date, df2.full_80,label="individuals who received full vaccination")
plt.plot(df2.date, df2.booster_80, label ="invidual who received 1st booster")
plt.legend(fontsize="15")
plt.show()


# In[33]:


# plot bar chart for children age 5-11 received full vaccination compare to different state
df2.groupby(pd.Grouper(key="state")).full_5_11.sum().plot(kind='bar')
plt.title ("Number of full vaccination age  between 5 -11 different state ", fontsize=15)
plt.xlabel("\nState",size=15)
plt.ylabel("Number of individuals who received vaccine \n",size=10)


# In[34]:


# plot bar chart for children age 12-11 received full vaccination compare to different state
df2.groupby(pd.Grouper(key="state")).full_12_17.sum().plot(kind='bar')
plt.title ("Number of full vaccination age  between 12-17 different state ", fontsize=15)
plt.xlabel("\nState",size=15)
plt.ylabel("Number of individuals who received vaccine \n",size=10)


# In[9]:


# Stage 4 : Data visualization using multiple bar
df3=df2.groupby(pd.Grouper(key="state")).sum()


# In[10]:


# to dsiplay total vaccination across the state
df3


# In[13]:


df3.index


# In[16]:


df3.info()


# In[13]:


# Multiple bar comparing vaccine and booster across age 60-69
import numpy as np
import matplotlib.pyplot as plt

X = df3.index
Y = df3["full_60_69"]
Z = df3["booster_60_69"]

X_axis = np.arange(len(X))

plt.bar(X_axis - 0.4, Y,-0.4, label = 'full_60_69')
plt.bar(X_axis + 0.2, Z,0.4, label = 'booster_60_69')

plt.xticks(X_axis, X,rotation=90)
plt.xlabel("State")
plt.ylabel("Number of individuals who recieved vaccine")
plt.title("Number of individuals who received vaccine according to state for age 60 -69")
plt.legend()
plt.show()


# In[14]:


# Multiple bar comparing vaccine and booster across age  70-79
import numpy as np
import matplotlib.pyplot as plt

X = df3.index
Y = df3["full_70_79"]
Z = df3["booster_70_79"]


X_axis = np.arange(len(X))

plt.bar(X_axis - 0.4, Y,0.4, label = 'full_70_79')
plt.bar(X_axis + 0.2, Z,0.4, label = 'booster_70_79')

plt.xticks(X_axis, X,rotation=90)
plt.xlabel("State")
plt.ylabel("Number of individuals who recieved vaccine")
plt.title("Number of individuals who received vaccine according to state age 70-79")
plt.legend()
plt.show()


# In[15]:


# Multiple bar comparing vaccine and bosster age > 80
import numpy as np
import matplotlib.pyplot as plt

X =df3.index
Y = df3["full_80"]
Z = df3["booster_80"]

X_axis = np.arange(len(X))

plt.bar(X_axis - 0.4, Y, 0.4, label = 'full_80')
plt.bar(X_axis + 0.2, Z, +0.4, label = 'booster_80')

plt.xticks(X_axis, X,rotation=90)
plt.xlabel("State")
plt.ylabel("Number of individuals who recieved vaccine")
plt.title("Number of individuals who received vaccine according to state for age >80")
plt.legend()
plt.show()


# In[11]:


# stage 5 gto check for correlation
df3.corr(method="pearson")


# In[ ]:


df3.skew


# In[ ]:


df3.kurt()

