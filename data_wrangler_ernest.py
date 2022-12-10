#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#data processing on covid_19 vax_malaysia.csv data


# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


#Importing csv file from github
data = pd.read_csv("https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/vaccination/vax_malaysia.csv")
data


# In[3]:


#checking the first 5 rows of data
data.head()


# In[4]:


#checking data types
data.dtypes


# In[5]:


#converting date object to datetime format
data['date'] = pd.to_datetime(data['date'])
data.dtypes


# In[6]:


#extracting variables of interest and rows up to 06-12-2022
data2 = data.iloc[0:651, 0:14]
data2


# In[18]:


pd.set_option('display.max_rows', None)
print(data2)


# In[19]:


#checking missing values
data2.isnull()


# In[8]:


#obtaining descriptive statistics on all variables of interest 
data2.describe()


# In[9]:


#calculating the total vaccinated individuals 
sum_column = data2.sum(axis=0)
print(sum_column)


# In[11]:


#storing the total vaccinated output in a data frame 
data3 = pd.DataFrame (sum_column, columns=['Total_vaccinated'])
data3


# In[10]:


#importing csv file containing total population information
pop_data = pd.read_csv("https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/static/population.csv")
pop_data


# In[12]:


#creating a dataframe containing information on total population based on vaccination age groups
pop = [32657100, 32657100, 32657100, 32675100, 32657100, 3111400,3111400,3111400,3111400,3550500,3550500,3550500,3550500]
data4 = pd.DataFrame(pop, columns=['Total_pop'], index=['daily_partial', 'daily_full', 'daily_booster', 'daily_booster2', 'daily', 'daily_partial_adol', 'daily_full_adol', 'daily_booster_adol', 'daily_booster2_adol', 'daily_partial_child', 'daily_full_child', 'daily_booster_child','daily_booster2_child'])
data4


# In[13]:


#combining data3 and data4 based on columns
data5 = pd.concat([data3, data4], axis="columns")
data5


# In[15]:


#calcculating percentage of individuals vaccinated in the population 
data5['Rate'] = data5['Total_vaccinated']/data5['Total_pop']*100
data5


# In[20]:


#plotting overall vaccine trend in Malaysia 

X = data["date"]
Y1 = data["daily_partial"]
Y2 = data["daily_full"]
Y3 = data["daily_booster"]
Y4 = data["daily_booster2"]

plt.figure(figsize=(25,8))
plt.title('Overall daily vaccine trend in Malaysia (24-02-2021 to 06-12-2022) \n', size=25)
plt.xlabel('\nDate', size=20)
plt.ylabel('Number of individuals who received daily vaccine \n', size=20)
plt.grid(linewidth=0.4, color='#8f8f8f')

plt.plot(X, Y1, color='blue', linewidth=2, label='Individuals who received 1st dose')
plt.plot(X, Y2, color='green', linewidth=2, label='Individuals who received 2nd dose')
plt.plot(X, Y3, color='fuchsia', linewidth=2, label='Individuals who received 1st booster')
plt.plot(X, Y4, color='red', linewidth=4, label='Individuals who received 2nd booster')
plt.legend(fontsize='18')
plt.show()


# In[21]:


#plotting overall vaccine trend for adolescents only

X = data["date"]

Y1 = data["daily_partial_adol"]

Y2 = data["daily_full_adol"]

Y3 = data["daily_booster_adol"]

Y4 = data["daily_booster2_adol"]

plt.figure(figsize=(25,8))
plt.title('Daily vaccine trend among adolescents (aged 12 - 17 only) in Malaysia (24-02-2021 to 06-12-2022) \n', size=25)
plt.xlabel('\nDate', size=20)
plt.ylabel('Number of individuals who received daily vaccine \n', size=20)
plt.grid(linewidth=0.4, color='#8f8f8f')

plt.plot(X, Y1, color='blue', linewidth=4, label='Individuals who received 1st dose')
plt.plot(X, Y2, color='green', linewidth=4, label='Individuals who received 2nd dose')
plt.plot(X, Y3, color='fuchsia', linewidth=4, label='Individuals who received 1st booster')
plt.plot(X, Y4, color='red', linewidth=4, label='Individuals who received 2nd booster')
plt.legend(fontsize='18')
plt.show()


# In[22]:


#plotting overall vaccine trend for children only

X = data["date"]

Y1 = data["daily_partial_child"]

Y2 = data["daily_full_child"]

Y3 = data["daily_booster_child"]

Y4 = data["daily_booster2_child"]

plt.figure(figsize=(25,8))
plt.title('Daily vaccine trend among children (aged 5 - 11 only) in Malaysia (24-02-2021 to 06-12-2022) \n', size=25)
plt.xlabel('\nDate', size=20)
plt.ylabel('Number of individuals who received daily vaccine \n', size=20)
plt.grid(linewidth=0.4, color='#8f8f8f')

plt.plot(X, Y1, color='blue', linewidth=4, label='Individuals who received 1st dose')
plt.plot(X, Y2, color='green', linewidth=4, label='Individuals who received 2nd dose')
plt.plot(X, Y3, color='fuchsia', linewidth=4, label='Individuals who received 1st booster')
plt.plot(X, Y4, color='red', linewidth=4, label='Individuals who received 2nd booster')
plt.legend(fontsize='18')
plt.show()


# In[23]:


#plotting overall cumulative vaccine trend 

X = data["date"]

Y1 = data["cumul_partial"]

Y2 = data["cumul_full"]

Y3 = data["cumul_booster"]

Y4 = data["cumul_booster2"]

plt.figure(figsize=(25,8))
plt.title('Cumulative vaccine trend in Malaysia (24-02-2021 to 06-12-2022) \n', size=25)
plt.xlabel('\nDate', size=20)
plt.ylabel('Cumulative number of individuals who received daily vaccine \n', size=20)
plt.grid(linewidth=0.4, color='#8f8f8f')

plt.plot(X, Y1, color='blue', linewidth=2, label='Individuals who received 1st dose')
plt.plot(X, Y2, color='green', linewidth=2, label='Individuals who received 2nd dose')
plt.plot(X, Y3, color='fuchsia', linewidth=2, label='Individuals who received 1st booster')
plt.plot(X, Y4, color='red', linewidth=4, label='Individuals who received 2nd booster')
plt.legend(fontsize='18')
plt.show()


# In[ ]:




