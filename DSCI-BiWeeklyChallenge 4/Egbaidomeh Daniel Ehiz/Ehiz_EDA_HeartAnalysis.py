#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# 

# In[2]:


ds = pd.read_csv(r"C:\Users\HP\Desktop\stutern_school\DSCI-BiWeeklyChallenge 4/heart_failure_clinical_records_dataset.csv")


# In[3]:


ds.head()
# ds.describe()


# In[4]:


print(ds.columns)


# #### Exploratory Analysis

# In[5]:


# Check the unique values in each of the columns
# ds['ejection_fraction'].unique()
# ds['time'].unique()
# ds['age'].unique()
# ds['serum_creatinine'].unique()
ds['anaemia'].unique()
ds['sex'].value_counts()


# In[6]:


# Age Classification
young_adult = list(range(45, 55))
mid_aged_adult = list(range(55, 65))
old_aged_adults = list(range(65, 75))
older_adults = list(range(75, 85))


# In[7]:


# Classify the age into bins
age_cut = pd.cut(ds['age'], bins = [39,55,65,75,100], labels= [0, 1, 2, 3])
ds.insert(0, 'age_class', age_cut)
ds.drop(['age'], axis = 1, inplace= True)


# In[8]:


ds


# In[9]:


ds.isnull().sum()


# In[10]:


plt.figure(figsize = (10, 25))

def plot_count(a,fig):
    plt.subplot(5,2,fig)
    plt.title(a+' vs sex')
    sns.countplot(x = 'sex', hue = a, data = ds )
    plt.subplot(5,2,(fig+1))
    plt.title(a+' vs age_class')
    sns.countplot(x = 'age_class', hue = a, data = ds )

plot_count('smoking', 1)
plot_count('DEATH_EVENT', 3)
plot_count('anaemia', 5)
plot_count('diabetes',  7)
plot_count('high_blood_pressure', 9)


# Findings in the graphical representation
# 
# Male = 1, Female = 0
# 
# There are more male smokers than female smokers. The young adults smokes hardest
# 
# More death cases in male than female, and the death occurs the most around 55 - 65 years of age
# 
# An average of 5 out of 10 females has anaemia. The mid aged adults (55 - 65 years) has the highest occurence of anaemia
# 
# An average of 6 out of 10 females has diabetes. Mid aged adults are more diabetic, while older adults has least occurence.
# 
# More females has High BP occurence than males. Old adults (65 -75 years) have higher occurence of High BP
# 

# In[11]:



plt.figure(figsize = (5, 20))
def death_count(a,fig):
    plt.subplot(4,1, fig)
    sns.countplot(x = 'DEATH_EVENT', hue = a, data = ds)
    plt.title('Count of '+ a +' Death Event')

death_count('high_blood_pressure', 1)
death_count('diabetes', 2)
death_count('anaemia', 3)
death_count('smoking', 4)
plt.plot()


# Findings in the graphical representation
# 
# Generally there are more survivors than dead patients. 7 out of every 10 patient survived, while 3 died.
# 
# There are lesser survivors of anaemia than any other listed ailment

# In[12]:


corr = ds.corr()
plt.figure(figsize=(15,10))
sns.heatmap(corr, annot = True, cmap= 'BuPu')


# No listed ailment causes another ailment, i.e, no causality between listed ailments

# 
