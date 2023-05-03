#!/usr/bin/env python
# coding: utf-8

# # Importance of data visualization

# To demonstrate the importance of data visualization for results interpretation we can use a collection of four datasets known as **Anscombe's quartet** (after an English statistician Francis Anscombe). Those datasets comprise 11 data points with particular properties: while the points themselves differ between the sets, their summary statistics are (nearly) exactly the same, i.e.: all those datasets have the same mean, standard deviation and regression line with the same parameters and R2 metric. And yet, the shape of the data differs widely between those, as we will see below.

# ## Environment setup
# 
# Here we will import all the required modules and preconfigure some variables, if necessary.

# In[1]:


import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy import stats

get_ipython().run_line_magic('config', "InlineBackend.figure_format='svg'")
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load the dataset and analyze it

# The *seaborn* library provides a method to read in this dataset directly:

# In[2]:


quartet = sns.load_dataset("anscombe")


# In[3]:


quartet.head() # display the first five records


# We can now explore the summary metrics of the data. Let's look at the mean, standard deviation and linear regression equation. To achieve this, we can use pandas' _groupby_ method which allows us to "group" the data points according to the indicated variable - in our case, "dataset". We can the use the resulting object to calculate some summary statistics on all the groups.

# In[4]:


quartet_grouped = quartet.groupby('dataset')


# In[5]:


quartet_grouped.mean()


# In[6]:


quartet_grouped.std()


# In[7]:


# fit a linear regression model for each dataset

for ds in quartet['dataset'].unique():
    dataset = quartet[quartet['dataset'] == ds]
    res = stats.linregress(dataset['x'], dataset['y'])
    print(
        f'Dataset {ds}:'
        f'\ty={round(res.slope, 3)}x+{round(res.intercept, 3)}'
        f'\tR-coeff={round(res.rvalue**2, 3)}'
        f'\tCorrelation-coeff={round(res.rvalue, 3)}'
    )


# As you can see, the numbers are nearly identical. Without visualizing the data it would be rather difficult to tell the datasets apart.
# 
# Let's try to look at the data using a scatter plot, including a regression line (automatically calculated by the `lmplot` function):

# In[8]:


with sns.plotting_context("notebook", font_scale=1.2), sns.axes_style('white'):
    g = sns.lmplot(
        x="x", y="y", col="dataset", data=quartet, ci=None,
        col_wrap=2, scatter_kws={"s": 100, 'color': '#0D62A3'},
        line_kws={'linewidth': 5, 'color': '#A30905', 'alpha': 0.5},
    )
    g.set(xlim=(2, 22))


# Now, that is interesting. Each of those looks entirely differently - let's think about how we could interpret each of those datasets after visual inspection:
# 
#  1. resembles a typical linear relationship where the y variable is correlated with x (with a lot of [Gaussian noise](https://en.wikipedia.org/wiki/Gaussian_noise))
#  2. there is a clear [correlation](https://en.wikipedia.org/wiki/Correlation) between variables but not a linear one
#  3. the relationship is evidently linear, with a single outlier (which is lowering the correlation coefficient from 1 to 0.816)
#  4. there does not seem to be a relationship between the two variables, however the outlier again skews the correlation coefficient significantly
# 
# As you can see, according to the linear regression model parameters themselves those datasets look very similar, if not the same. It is only when we visualize them using a simple scatter plot we can see that those datasets differ significantly.
# 
# ````{admonition} Remember
# :class: tip
# Data visualization allows us to get the first glimpse into the properties of the dataset, without the need to calculate many different summary statistics and other metrics. 
# ````

# In the next sections, we will explore different visualization methods that will help us in the analysis of different datasets.
