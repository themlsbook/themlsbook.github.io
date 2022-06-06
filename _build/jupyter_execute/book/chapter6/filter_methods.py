#!/usr/bin/env python
# coding: utf-8

# (chapter6_part1)=
# 
# # Filter Methods
# 
# - This is a supplement material for the [Machine Learning Simplified](https://themlsbook.com) book. It sheds light on Python implementations of the topics discussed while all detailed explanations can be found in the book. 
# - I also assume you know Python syntax and how it works. If you don't, I highly recommend you to take a break and get introduced to the language before going forward with my code. 
# - This material can be downloaded as a Jupyter notebook (Download button in the upper-right corner -> `.ipynb`) to reproduce the code and play around with it. 
# 
# 
# This notebook is a supplement for *Chapter 3. Dimensionality Reduction Techniques* of **Machine Learning For Everyone** book.
# 
# ## 1. Required Libraries, Data & Variables
# 
# Let's import the data and have a look at it:

# In[1]:


import pandas as pd

data = pd.read_csv('https://github.com/5x12/themlsbook/raw/master/supplements/data/car_price.csv', delimiter=',', header=0)


# In[2]:


data.head()


# In[3]:


data.columns


# Let's define features $X$ and a target variable $y$:

# In[4]:


data['price']=data['price'].astype('int')

X = data[['wheelbase', 
          'carlength', 
          'carwidth', 
          'carheight', 
          'curbweight', 
          'enginesize', 
          'boreratio', 
          'stroke',
          'compressionratio', 
          'horsepower', 
          'peakrpm', 
          'citympg', 
          'highwaympg']]

y = data['price']


# Let's split the data:

# In[5]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# ## 2. Filter methods
# 
# The following Filter methods are examined:
# 
#    1. **Chi Square** method
#    2. **Fisher Score** method
#    3. **RelieF** method
#    4. **Correlation-based** Feature Selection method
# 
# ### 2.1. Chi-square

# In[6]:


# Importing required libraries
from sklearn.feature_selection import chi2


# In[7]:


# Set and fit Chi-square feature selection
chi = chi2(X_train, y_train)


# In[8]:


chi


# In[9]:


# Create a list with feature label and its p-value
chi_features = pd.Series(chi[1], index = X_train.columns) # create a series with feature labels and their corresponding p-values
chi_features.sort_values(ascending = True, inplace = True) # sort series by p-values


# In[10]:


# Return features with p-values
chi_features


# In[11]:


# Print 4 best features
chi_features[:4]


# In[12]:


# Print features whose p-value < 0.05
for feature_name, feature_score in zip(X.columns,chi[1]):
    if feature_score<0.05:
        print(feature_name, '\t', feature_score)


# ### 2.2. Fisher Score

# In[13]:


# Importing required libraries
from skfeature.function.similarity_based import fisher_score


# In[14]:


# Set Fisher Score
score = fisher_score.fisher_score(X_train.values, y_train.values)
score


# In[15]:


# Create a list with feature label and its p-value
f_values = pd.Series(score, index = X_train.columns) # create a series with feature labels and their corresponding fisher scores
f_values.sort_values(ascending = True, inplace = True) # sort series by fisher score


# In[16]:


f_values


# ### 2.3. RelieF

# In[17]:


# Importing required libraries
get_ipython().system(' pip install ReliefF')
from ReliefF import ReliefF


# In[18]:


# Set ReliefF method
fs = ReliefF(n_neighbors=1, n_features_to_keep=4)

# Perform ReliefF by fitting X and y values
fs.fit_transform(X_train.values, y_train.values)

# Make a ranking list with feature scores
relief_values = pd.Series(fs.feature_scores, index = X_train.columns) # create a series with feature labels and their corresponding ReliefF scores
relief_values.sort_values(ascending = True, inplace = True) # sort series by ReliefF score
relief_values


# When using original Relief or ReliefF, it has been suggested that features yielding a negative value score, can be confidently filtered out. Now, feature $horsepower$ is negative, which implies it is redundant. With some commonsense knowledge, we know the horsepower is one of the strongest parameters affecting the price of a car. That's why you should be careful when applying this feature selection technique. The best way out is to try out several feature selection methods to see the general pattern.

# In[19]:


# Print a ranking list with top 5 features
relief_features = []
for feature_name, feature_score in zip(X.columns,fs.feature_scores):
    if feature_score>15:
        relief_features.append(feature_name)
        print(feature_name, '\t', feature_score)


# In[20]:


# Selected features that satisfy criteria
relief_features


# ### 2.4. Correlation-based Feature Selection

# In[21]:


#Correlation with output variable
cor = data.corr()
cor_target = abs(cor['price'])

#Selecting highly correlated features > 0.8
relevant_features = cor_target[:-1][cor_target>0.8]
relevant_features


# ## 3. Comparing Four Methods

# In[22]:


print('The features selected by chi-square are: \n \n {} \n \n \n The features selected by f_values are: \n \n {} \n \n \n The features selected by ReliefF are: \n \n {} \n \n \n The features selected by Correlation-based feature selection method are: \n \n {}'.format(chi_features, f_values, relief_features, relevant_features))


