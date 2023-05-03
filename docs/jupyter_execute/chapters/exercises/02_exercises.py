#!/usr/bin/env python
# coding: utf-8

# # Exercises

# ## Environment setup

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# this is to silence pandas' warnings
import warnings
warnings.simplefilter(action='ignore')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='svg'")

FONT_FAMILY = 'DejaVu Sans'
FONT_SCALE = 1.3

data_dir = '../data'


# ## Data pre-processing
# 
# All the exercises are based on the Kaggle [**cereals**](https://www.kaggle.com/code/hiralmshah/nutrition-data-analysis-from-80-cereals) dataset, which has 77 records and 16 columns containing nutritional information on different brands of breakfast cereals. The columns are:
# - **name** - name of the cereal
# - **mfr** - manufacturer of the cereals. You can find the association of the letter in the dataset with the real name in the `manufacturers_df` we have loaded below.
# - **type** - hot or cold, the preferred way of eating
# - **calories** - amount of calories
# - **fat** - grams of fat
# - **sodium** - milligrams of sodium
# - **fiber** - amount in grams
# - **carbo** - amount of carbohydrates in grams
# - **sugars** - amount in gram
# - **potass** - amount in milligrams
# - **vitamins** - vitamins and minerals (0, 25, 100) as a percentage of the Recommended Dietary Intake
# - **shelf** - shelf they appear in supermarket (1, 2 or 3 from the floor)
# - **weight** - weight in ounces
# - **cups** - number of cups
# - **rating** - rating of the cereals
# 
# ````{admonition} Note
# All the values are expresed per 100g portion.
# ````

# In[2]:


# load main dataset
cereals_df = pd.read_csv(f'{data_dir}/cereal.csv', sep=',')
cereals_df.head()


# In[3]:


# load dataset that maps manufacturer letter codes to their names
manufacturers_df = pd.read_csv(f'{data_dir}/manufacturers.csv', index_col=0)
manufacturers_df


# In[4]:


# merge the two datasets
cereals = pd.merge(
    cereals_df, manufacturers_df, left_on=cereals_df.mfr, right_index=True
)
# remove duplicated column
cereals.drop('key_0', axis=1, inplace=True)
cereals.head()


# ## Exercises

# ### Exercise 1
# Plot the **number of products per manufacturer** by displaying the manufacturer's name instead of the letter that appears in the `cereals_df` dataframe. All the data you need is found in the `cereals` DataFrame.

# In[5]:


# write your code here


# ### Exercise 2
# Plot the **distribution of ratings per company** checking at the same time if there are any **outliers**. You can find the necessary data in the `data` DataFrame.

# In[6]:


data = cereals[['company_name', 'rating']]


# In[7]:


# write your code here


# ### Exercise 3
# Find and visualize the **ratings per product**. You will find the necessary data in the `data` DataFrame.

# In[8]:


data = cereals[['name', 'rating']].groupby('name').mean().reset_index()


# In[9]:


# write your code here


# ### Exercise 4
# Find if there is a **correlation between any of the numerical features** we have in the dataset. Again, you will find the data needed in the `data` DataFrame.

# In[10]:


data = cereals[['calories', 'protein', 'fat', 'sodium', 'fiber', 'carbo', 'sugars', 'potass', 'rating']]


# In[11]:


# write your code here


# ### Exercise 5
# Your next task is to find and visualize these correlations in a more **quantitative** way. The data will be ready for you in the `data` dataframe, you will only have to find the correct visualization method and supply the correct arguments to the function.

# In[12]:


data = cereals[['fiber', 'potass', 'sugars', 'calories','rating']]


# In[13]:


# write your code here


# ### Exercise 6
# Using a scatterplot, show how the **potassium amount changes w.r.t. the fiber amount and the rating**. Notice that this requires you to plot three numerical variables at the same time. The data to be used is ready for you in the `data` DataFrame.

# In[14]:


data = cereals[['potass', 'fiber', 'rating']]


# In[15]:


# write your code here


# ### Exercise 7
# Using a scatterplot, plot **the potassium amount w.r.t. to the fiber amount, the sugar amount and the rating**. Notice that this will required you to find a visualization allowing to display four variables at once. The data to be used is ready for you in the `data` DataFrame. 
# 
# You might find some useful information [here](https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_with_legend.html#sphx-glr-gallery-lines-bars-and-markers-scatter-with-legend-py) and [here](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html#matplotlib.pyplot.scatter).

# In[16]:


data = cereals[['potass', 'fiber', 'sugars', 'rating']]


# In[17]:


# write your code here

