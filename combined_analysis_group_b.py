#!/usr/bin/env python
# coding: utf-8


### Ernest Section
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

#### END OF Ernest Section


# In[1]:

#### Irfan Section

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

# In[ ]:

#### End of Irfan Section



#### Isaac Section

###---------------
# Project HIA302 Group B 
# Isaac Kong
# Adverse Event Following Immunization (aefi) Trends 
###----------------


# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import os, sys


# In[2]:


get_ipython().run_cell_magic('bash', '', '\n### Cloning database directly from github MOH-Malaysia/covid19-public\n# git clone https://github.com/MoH-Malaysia/covid19-public.git\n### Remove .git after cloning to avoid having a repo inside a repo\n# rm -rf covid19-public/.git ')


# In[3]:


#create data path to access data file from path 
data_path="covid19-public/vaccination/"

# to show the all files in the data_path 
vax_files = os.listdir(data_path)
vax_files


# In[4]:


pd.read_csv(os.path.join(data_path, "aefi.csv")).describe()


# In[5]:


# identify the extension (file type)
filename, filext = os.path.splitext(vax_files[8])

filext

if(filext == '.csv'):
    print('this is a CSV')
else:
    print('this is not a csv file')
filename


# In[6]:


# to read the dataframe for aefi 
pd.read_csv(os.path.join(data_path, "aefi.csv"))


# In[7]:


# to read the dataframe for aefi_serious 
pd.read_csv(os.path.join(data_path, "aefi_serious.csv"))


# In[8]:


### Create data repository holiding all datasets for vacincations 

data_repo = dict()

for file in vax_files:
    file_path = os.path.join(data_path, file)
    filename, filext = os.path.splitext(file)

    if(filext == ".csv"):
        print(file_path)
        tmp_df = pd.read_csv(file_path)
        data_repo[filename] = tmp_df.copy()
        
data_repo.keys()


# In[9]:


# Perform data check to identify missing value or data completeness 
aefi = data_repo['aefi'].copy()

aefi_serious = data_repo['aefi_serious'].copy()

print(aefi.info())
print(aefi_serious.info())


# In[10]:


# Merging of 2 dataframes (aefi & aefi_serious) into one by 'date' and 'vaxtype'
aefi_full = aefi_serious.merge(aefi, on=['date', 'vaxtype'], how='right')
aefi_full


# In[11]:


#data wrangling step 1 for dose 1

### Select only columns that start with 'd1'
filter_col_d1 = list(aefi_full.columns[aefi_full.columns.str.startswith("d1_")])

### Create a datafrane that has only date, vaxtype, and columns starting with d1
aefi_dose1_final = aefi_full.loc[: ,["date" , "vaxtype"]+ filter_col_d1]
aefi_dose1_final


# In[12]:


#data wrangling step 1 for dose 2

### Select columns that start with 'd2'
filter_col_d2 = list(aefi_full.columns[aefi_full.columns.str.startswith("d2_")])

### Create a datafrane that has only date, vaxtype, and columns starting with d2
aefi_dose2_final = aefi_full.loc[: ,["date" , "vaxtype"]+ filter_col_d2]
aefi_dose2_final


# In[13]:


aefi_dose2_final.info()


# In[14]:


# data rearrangement Unpivot a DataFrame from wide to long format using pandas (pd.wide_to_long) for dose 1

# dose 1
aefi_dose1_l = pd.wide_to_long(aefi_dose1_final, stubnames='d1', i=['date', 'vaxtype'], j='AE', sep="_", suffix=r'\w+')
aefi_dose1_l.rename(columns = {'d1':'count'}, inplace = True)
# 
aefi_dose1_l["dose"]=1
aefi_dose1_l


# In[15]:


aefi_dose1_l[(aefi_dose1_l['count']>0)]


# In[16]:



tmp = aefi_dose1_final.groupby(['date','vaxtype']).sum()

tmp_stack = pd.DataFrame(tmp.stack()).rename(columns={0: 'count'})

tmp_stack.index.names = ['date', 'vaxtype', 'AE']
tmp_stack['dose'] = 1 

tmp_stack


# In[17]:


tmp_stack


# In[18]:


# data rearrangement Unpivot a DataFrame from wide to long format using pandas (pd.wide_to_long) for dose 2

# dose 2
aefi_dose2_l = pd.wide_to_long(aefi_dose2_final, stubnames='d2', i=['date', 'vaxtype'], j='AE', sep="_", suffix=r'\w+')
aefi_dose2_l.rename(columns = {'d2':'count'}, inplace = True)
# 
aefi_dose2_l["dose"]=2
aefi_dose2_l


# In[19]:


# dose1 AEFI Pie chart, where the slices will be ordered and plotted counter-clockwise:
aefi_dose1_summ = aefi_dose1_l.groupby(['AE'])['count'].sum().nlargest(10)
aefi_dose1_summ.plot(kind='pie', y='count', figsize=(50,10), autopct='%1.0f%%',
                     title = 'Top 10 Adverse Events Following Immunization on Dose 1' )


# In[20]:


# dose1 bar chart, where the slices will be ordered and plotted counter-clockwise:
aefi_dose1_summ = aefi_dose1_l.groupby(['AE'])['count'].sum().nlargest(10)
aefi_dose1_summ_plt_bar = aefi_dose1_summ.plot(kind='bar', stacked=True,figsize=(20,12),
                                               title = 'Top 10 Adverse Events Following Immunization on Dose 1',
                                               color = {"darkorange"})

for container in aefi_dose1_summ_plt_bar.containers:
    aefi_dose1_summ_plt_bar.bar_label(container)


# In[21]:


# dose 2 Pie chart, where the slices will be ordered and plotted counter-clockwise:
aefi_dose2_summ = aefi_dose2_l.groupby(['AE'])['count'].sum().nlargest(10)
aefi_dose2_summ.plot(kind='pie', y='count', autopct='%1.0f%%',figsize=(50,10),
                     title = 'Top 10 Adverse Events Following Immunization on Dose 1')


# In[22]:


# dose2 bar chart, where the slices will be ordered and plotted counter-clockwise:
aefi_dose2_summ = aefi_dose2_l.groupby(['AE'])['count'].sum().nlargest(10)
plt = aefi_dose2_summ.plot(kind='bar', stacked=True,figsize=(20,12), 
                           title = 'Top 10 Adverse Events Following Immunization on Dose 2', 
                           color = "darkblue")

for container in plt.containers:
    plt.bar_label(container)


# In[23]:


# dose 1 concatinate with dose 2 dataframe
aefi_dose12_final = pd.concat([aefi_dose1_l,aefi_dose2_l],axis=0)
aefi_dose12_final


# In[24]:


# barchart by vaccine brand
vaxtype_summ_ae = aefi_dose12_final.groupby(['vaxtype', 'dose'])['count'].sum().unstack().plot(kind='bar', stacked=False ,figsize=(25,10), 
                                                                                               title = 'Number of Adverse Event Following Immumnization by Vaccine Type & Doses')

plt_2 = vaxtype_summ_ae

for container in plt_2.containers:
    plt_2.bar_label(container, fmt="%1d")


# In[26]:


# regardless of dose (SAE inclusive) Pie chart, where the slices will be ordered and plotted counter-clockwise:
aefi_dose12_summ = aefi_dose12_final.groupby(['AE'])['count'].sum().nlargest(10)
aefi_dose12_summ.plot(kind='pie', y='count', autopct='%1.0f%%',figsize=(50,10))


# In[27]:


# Serious AE 
# aefi_serious = aefi_full[['date','vaxtype','suspected_anaphylaxis','acute_facial_paralysis','venous_thromboembolism', 'myo_pericarditis']]
aefi_serious
aefi_serious_l = pd.melt(aefi_serious, id_vars= ['date','vaxtype'], value_vars=['suspected_anaphylaxis','acute_facial_paralysis','venous_thromboembolism', 'myo_pericarditis'])

aefi_serious_l.rename(columns = {'value':'count','variable':'AE'}, inplace = True)
aefi_serious_l

aefi_serious_l_summ_pie = aefi_serious_l.groupby(['AE'])['count'].sum()
aefi_serious_l_summ_pie.plot(kind='pie', y='count', autopct='%1.0f%%',figsize=(50,10))


# In[28]:


# Serious AE 
# aefi_serious = aefi_full[['date','vaxtype','suspected_anaphylaxis','acute_facial_paralysis','venous_thromboembolism', 'myo_pericarditis']]
aefi_serious
aefi_serious_l = pd.melt(aefi_serious, id_vars= ['date','vaxtype'], value_vars=['suspected_anaphylaxis','acute_facial_paralysis','venous_thromboembolism', 'myo_pericarditis'])

aefi_serious_l.rename(columns = {'value':'count','variable':'AE'}, inplace = True)
aefi_serious_l

aefi_serious_l_summ_bar = aefi_serious_l.groupby(['AE'])['count'].sum()
aefi_serious_l_summ_bar_plt = aefi_serious_l_summ_bar.plot(kind='bar', stacked=True, figsize=(15,5), 
                                                           title = 'Serious Adverse Events Following Immunization Regardless of Dose', 
                                                           color = "indigo")

for container in aefi_serious_l_summ_bar_plt.containers:
    aefi_serious_l_summ_bar_plt.bar_label(container)


# In[29]:


#### Analysis to compare total vaxed vs Adverse Events Following Immunization (AEFI)
### Datasets used:
### Vax Malaysia: Contains all number of vaxcinations (dose 1 and dose 2)
### AEFI: contains all adverse events after vax dose 1 and dose 2
### data_repo contains all datasets required


# In[30]:


### AEFI stacked by doses dataset

aefi_dose12_final2 = aefi_dose12_final.reset_index()

## Convert date to datetime data type
aefi_dose12_final2['date'] = pd.to_datetime(aefi_dose12_final2['date'])

### For bar plots, transform dates into a format of Year-Month for better plotting
aefi_dose12_final2['date_shrt'] = pd.to_datetime(aefi_dose12_final2['date']).dt.strftime('%Y-%m')
aefi_dose12_final2


# In[31]:


# Vax Malaysia data

vax_my = data_repo['vax_malaysia'].copy()

## Convert date to datetime data type
vax_my['date'] = pd.to_datetime(vax_my['date'])

### For bar plots, transform dates into a format of Year-Month for better plotting
vax_my['date_shrt'] = pd.to_datetime(vax_my['date']).dt.strftime('%Y-%m')
vax_my


# In[32]:


### Describe dist of dates in Vax Malaysia
vax_my['date'].describe()


# In[33]:


### Describe dist of dates in AEFI

aefi_dose12_final2['date'].describe()


# In[34]:


### Define the overlap period of time between Vax Malaysia data and AEFI datasets 
### in order to compare total vaxed vs adverse events within the same period of time

### Take the latest date from the earliest dates of Vax Malaysia and AEFI, that's the start date 
start_dt = np.max([vax_my['date'].min(), aefi_dose12_final2['date'].min()])

### Take the earliest date from the latest dates of Vax Malaysia and AEFI, that's the end date
end_dt = np.min([vax_my['date'].max(), aefi_dose12_final2['date'].max()])

print("start date: ", start_dt)
print("end date: ", end_dt)


# In[35]:


### Apply filter to only include records between start date and end date
vax_my_flt = vax_my[((vax_my['date'] >= start_dt) & (vax_my['date'] <= end_dt))].copy()
print(vax_my_flt.shape)
vax_my_flt.head()


# In[36]:


### Create a dataset stacked by full date (year, month, day) for distribtuion plot
vax_my_dist = vax_my_flt[[ 'date', 'daily_partial', 'daily_full']].groupby('date').sum().reset_index()

vax_my_dist_dose1 = vax_my_dist[['date', 'daily_partial']].rename(columns={'daily_partial': 'count'})
vax_my_dist_dose1['dose'] = 1
vax_my_dist_dose2 = vax_my_dist[['date', 'daily_full']].rename(columns={'daily_full': 'count'})
vax_my_dist_dose2['dose'] = 2

vax_my_dist_stacked  = pd.concat([vax_my_dist_dose1, vax_my_dist_dose2])

vax_my_dist_stacked.head()


# In[37]:


### Create a dataset staacked by full date (year, month) for Bar plot

vax_my_tmp = vax_my_flt[[ 'date_shrt', 'daily_partial', 'daily_full']].groupby('date_shrt').sum().reset_index()

vax_my_tmp_dose1 = vax_my_tmp[['date_shrt', 'daily_partial']].rename(columns={'daily_partial': 'count'})
vax_my_tmp_dose1['dose'] = 1
vax_my_tmp_dose2 = vax_my_tmp[['date_shrt', 'daily_full']].rename(columns={'daily_full': 'count'})
vax_my_tmp_dose2['dose'] = 2

vax_my_stacked  = pd.concat([vax_my_tmp_dose1, vax_my_tmp_dose2])

vax_my_stacked.head()


# In[38]:


### Apply filter to only include records between start date and end date


aefi_dose12_final2_flt = aefi_dose12_final2[((aefi_dose12_final2['date'] >= start_dt) & (aefi_dose12_final2['date'] <= end_dt))].copy()

aefi_dose12_final2_flt_stacked = aefi_dose12_final2_flt[['date_shrt', 'dose', 'count']].groupby(['date_shrt', 'dose' ]).sum().reset_index()
print(aefi_dose12_final2_flt_stacked.shape)
aefi_dose12_final2_flt_stacked.head()


# In[39]:


### Create a dataset staacked by full date (year, month, day) for distribution plot

aefi_dose12_final2_flt_dist_stacked = aefi_dose12_final2_flt[['date', 'dose', 'count']].groupby(['date', 'dose' ]).sum().reset_index()
aefi_dose12_final2_flt_dist_stacked.head()


# In[40]:


### Combine Vax malaysia stacked dataset with AEFI stacked dataset for Bar plots

vax_my_stacked['group'] = 'Total Vaxed' 
aefi_dose12_final2_flt_stacked['group'] = 'AE'
total_vax_nd_ae_stacked = pd.concat([vax_my_stacked, aefi_dose12_final2_flt_stacked])

total_vax_nd_ae_stacked.groupby(['date_shrt','group', 'dose']).sum()


# In[41]:


### Total number of events by dose and either Total Vax vs AE

total_vax_nd_ae_stacked.drop('date_shrt', axis=1).groupby(['group', 'dose']).sum()


# In[42]:


### Combine Vax malaysia stacked dataset with AEFI stacked dataset for distribution plots


vax_my_dist_stacked['group'] = 'Total Vaxed' 
aefi_dose12_final2_flt_dist_stacked['group'] = 'AE'
total_vax_nd_ae_dist_stacked = pd.concat([vax_my_dist_stacked, aefi_dose12_final2_flt_dist_stacked])

total_vax_nd_ae_dist_stacked.groupby(['date','group', 'dose']).sum()


# In[44]:


### Plotting distribtion plot showing trends of AEFI vs Total Vax over the study period 
### 2021-02-24 ==> 2022-06-07 
import matplotlib.pyplot as plt

aefi_by_time = total_vax_nd_ae_dist_stacked.groupby(['date','group'])['count'].sum().unstack().plot(figsize=(15,10),  title = 'Total Vaxed vs AE Events Distribution')
aefi_by_time


# plt.show()
plt.savefig('total_vax_vs_aefi_by_dose_dist.jpg')


# In[45]:


### Plotting barplot showing trends of Total Vaxed vd AEFI for dose 1 over the study period 
### 2021-02-24 ==> 2022-06-07 
total_vax_nd_ae_stacked_dose1 = total_vax_nd_ae_stacked.loc[ total_vax_nd_ae_stacked['dose']==1, ['date_shrt', 'group', 'count']].copy()


plt_tmp = total_vax_nd_ae_stacked_dose1.groupby(['date_shrt', 'group'])['count'].sum().unstack().plot(kind='bar',width=1, stacked=False, figsize=(15,10), 
                                                                                         title = 'Barplot of Total Vaxed vs AEFI For Dose 1')
plt_tmp

plt.savefig('total_vax_vs_aefi_for_dose1_barplot.jpg')


# In[46]:


### Plotting barplot showing trends of Total Vaxed vd AEFI for dose 2 over the study period 
### 2021-02-24 ==> 2022-06-07 
total_vax_nd_ae_stacked_dose2 = total_vax_nd_ae_stacked.loc[ total_vax_nd_ae_stacked['dose']==2, ['date_shrt', 'group', 'count']].copy()


# trend of adverse event distribution pattern overtime by dosing 
plt_tmp = total_vax_nd_ae_stacked_dose2.groupby(['date_shrt', 'group'])['count'].sum().unstack().plot(kind='bar',width=1, stacked=False, figsize=(15,10), 
                                                                                         title = 'Barplot of Total Vaxed vs AEFI For Dose 2')
plt_tmp

plt.savefig('total_vax_vs_aefi_for_dose2_barplot.jpg')


# In[47]:


### Plotting barplot showing aggregated sum of total vaxed vs AEFI by dose over the study period 
### 2021-02-24 ==> 2022-06-07 
plt_tmp = total_vax_nd_ae_stacked.groupby(['dose', 'group'])['count'].sum().unstack().plot(kind='bar',width=0.4, stacked=False, figsize=(12,9), 
                                                                                         title = 'Aggregate Sum of Total Vaxed vs AEFI By Dose')
plt_tmp

for container in plt_tmp.containers:
    plt_tmp.bar_label(container, fmt="%1d")


plt.savefig('agg_total_vax_vs_aefi_by_dose_barplot.jpg')

#### End of Isaac Section
