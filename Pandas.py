
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


dataset_file = R".\Data\populationbycountry19802010millions.csv"


# In[8]:


df_pop = pd.read_csv(dataset_file)
print(df_pop.head())


# In[30]:


print(df_pop.tail())   # Prints the last 5 rows of the data frame


# In[27]:


print(df_pop)


# In[5]:


print(type(df_pop))  # <class 'pandas.core.frame.DataFrame'>


# In[10]:


print(df_pop.shape) # (232, 32) This is a tuple  -> 232 countries, 32 years worth data


# In[11]:


print(df_pop.columns)


# In[13]:


print(df_pop.dtypes)   # Print the data type of the elements in each column. 
                       # All elements of a column can have only one specific type of data
                       # Everything in python is an object. Also, strings are considered object types in Pandas 


# In[14]:


print(df_pop.info())  # To get some summary of the data frame


# We can extract information based on the column names:

# In[16]:


pop_1980 = df_pop['1980']   # Extract all the columns for the year 1980 into a series


# In[19]:


print(type(pop_1980))   # <class 'pandas.core.series.Series'>


# In[29]:


print(pop_1980)


# In[37]:


print(df_pop.iloc[0][0])


# In[41]:


df_pop_1980_83 = df_pop[['Unnamed: 0', '1980', '1981', '1982', '1983']]  
# Get the columns from 1980 to 1983 along with the country name


# In[43]:


print(df_pop_1980_83.head())


# Since we don't have a proper name for the countries column, let us rename the same

# In[44]:


df_pop = df_pop.rename(index = {'Unnamed: 0': 'Country'}, inplace = True)


# In[47]:


print(df_pop.head())

