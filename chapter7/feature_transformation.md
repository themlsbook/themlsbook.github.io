---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(chapter7_part2)=

# Feature Transformation & Binning

- This is a supplement material for the [Machine Learning Simplified](https://themlsbook.com) book. It sheds light on Python implementations of the topics discussed while all detailed explanations can be found in the book. 
- I also assume you know Python syntax and how it works. If you don't, I highly recommend you to take a break and get introduced to the language before going forward with my code. 
- This material can be downloaded as a Jupyter notebook (Download button in the upper-right corner -> `.ipynb`) to reproduce the code and play around with it. 


This notebook is a supplement for *Chapter 7. Data Preparation* of **Machine Learning For Everyone** book.

## 1. Required Libraries, Data & Variables

Let's import the data and have a look at it:


```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler


# Define the data as a dictionary
data = {
    "Age": [32, 46, 25, 36, 29, 54],
    "Income (€)": [95000, 210000, 75000, 30000, 55000, 430000],
    "Vehicle": ["none", "car", "truck", "car", "none", "car"],
    "Kids": ["no", "yes", "yes", "yes", "no", "yes"],
    "Residence": ["downtown", "downtown", "suburbs", "suburbs", "suburbs", "downtown"]
}

# Create DataFrame from the dictionary
df = pd.DataFrame(data)

# Print the DataFrame
df

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Income (€)</th>
      <th>Vehicle</th>
      <th>Kids</th>
      <th>Residence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>32</td>
      <td>95000</td>
      <td>none</td>
      <td>no</td>
      <td>downtown</td>
    </tr>
    <tr>
      <th>1</th>
      <td>46</td>
      <td>210000</td>
      <td>car</td>
      <td>yes</td>
      <td>downtown</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25</td>
      <td>75000</td>
      <td>truck</td>
      <td>yes</td>
      <td>suburbs</td>
    </tr>
    <tr>
      <th>3</th>
      <td>36</td>
      <td>30000</td>
      <td>car</td>
      <td>yes</td>
      <td>suburbs</td>
    </tr>
    <tr>
      <th>4</th>
      <td>29</td>
      <td>55000</td>
      <td>none</td>
      <td>no</td>
      <td>suburbs</td>
    </tr>
    <tr>
      <th>5</th>
      <td>54</td>
      <td>430000</td>
      <td>car</td>
      <td>yes</td>
      <td>downtown</td>
    </tr>
  </tbody>
</table>
</div>



## 2. Feature Encoding

After having cleaned your data, you must encode it in a way such that the ML algorithm can consume it.One important thing you must do is encode complex data types, like strings or categorical variables, in a numeric format.

We will illustrate feature encoding on the dataset above, where the  three independent variables are Income, Vehicle, and Kids, each of which are categorical variables, and the target variable is a person's Residence (whether a person lives in downtown or in suburbs).

### 2.1. Apply One-Hot Encoding to "Vehicle" and "Kids"


```python
# One-hot encode the 'Vehicle' and 'Kids' columns
ohe = OneHotEncoder(sparse=False)
encoded_features = pd.DataFrame(ohe.fit_transform(df[['Vehicle', 'Kids']]))

# Get new column names from OneHotEncoder
encoded_features.columns = ohe.get_feature_names_out(['Vehicle', 'Kids'])

# Concatenate the encoded features back to the original DataFrame
df_encoded = pd.concat([df, encoded_features], axis=1).drop(['Vehicle', 'Kids'], axis=1)
df_encoded
```

    /Users/andrewwolf/.pyenv/versions/3.10.7/lib/python3.10/site-packages/sklearn/preprocessing/_encoders.py:808: FutureWarning: `sparse` was renamed to `sparse_output` in version 1.2 and will be removed in 1.4. `sparse_output` is ignored unless you leave `sparse` to its default value.
      warnings.warn(





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Income (€)</th>
      <th>Residence</th>
      <th>Vehicle_car</th>
      <th>Vehicle_none</th>
      <th>Vehicle_truck</th>
      <th>Kids_no</th>
      <th>Kids_yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>32</td>
      <td>95000</td>
      <td>downtown</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>46</td>
      <td>210000</td>
      <td>downtown</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25</td>
      <td>75000</td>
      <td>suburbs</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>36</td>
      <td>30000</td>
      <td>suburbs</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>29</td>
      <td>55000</td>
      <td>suburbs</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>54</td>
      <td>430000</td>
      <td>downtown</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



### 2.2. Apply Label Encoding to "Residence"


```python
# Label encode the 'Residence' column
le = LabelEncoder()
df_encoded['Residence'] = le.fit_transform(df_encoded['Residence'])
```


```python
df_encoded
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Income (€)</th>
      <th>Residence</th>
      <th>Vehicle_car</th>
      <th>Vehicle_none</th>
      <th>Vehicle_truck</th>
      <th>Kids_no</th>
      <th>Kids_yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>32</td>
      <td>95000</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>46</td>
      <td>210000</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25</td>
      <td>75000</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>36</td>
      <td>30000</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>29</td>
      <td>55000</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>54</td>
      <td>430000</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



## 3. Feature Scaling

Many datasets contain numeric features with significantly different numeric scales.

For example, the Age feature ranges from 27 to 54 (years), while the Income feature ranges from $30,000$ EUR to $430,000$ EUR, while the features Vehicle\_none, Vehicle\_car, Vehicle\_truck, Kids\_yes and Kids\_no all have the range from $0$ to $1$.

Unscaled data will, technically, not prohibit the ML algorithm from running, but can often lead to problems in the learning algorithm.

For example, since the Income feature has much larger value than the other features, it will influence the target variable much more.\footnote{This is the case for many ML models, such as linear models we discussed in the last section.  However, some ML models like decision trees are invariant to feature scaling.

But we don't necessarily want this to be the case.

To ensure that the measurement scale doesn't adversely affect our learning algorithm, we scale, or normalize, each feature to a common range of values. Here is an example Python code that demonstrates how to scale your features using `StandardScaler`:


```python
# Initialize the StandardScaler
scaler = StandardScaler()
```


```python
# List of all the columns
df_encoded.columns
```




    Index(['Age', 'Income (€)', 'Residence', 'Vehicle_car', 'Vehicle_none',
           'Vehicle_truck', 'Kids_no', 'Kids_yes'],
          dtype='object')




```python
# List of columns to scale (we take all but| target variable)
columns_to_scale = ['Age', 'Income (€)', 'Vehicle_car', 'Vehicle_none',
       'Vehicle_truck', 'Kids_no', 'Kids_yes']
```


```python
# Fit the scaler to the data and transform it
df_encoded[columns_to_scale] = scaler.fit_transform(df_encoded[columns_to_scale])
```


```python
df_encoded
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Income (€)</th>
      <th>Residence</th>
      <th>Vehicle_car</th>
      <th>Vehicle_none</th>
      <th>Vehicle_truck</th>
      <th>Kids_no</th>
      <th>Kids_yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.498342</td>
      <td>-0.392844</td>
      <td>0</td>
      <td>-1.0</td>
      <td>1.414214</td>
      <td>-0.447214</td>
      <td>1.414214</td>
      <td>-1.414214</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.897015</td>
      <td>0.441194</td>
      <td>0</td>
      <td>1.0</td>
      <td>-0.707107</td>
      <td>-0.447214</td>
      <td>-0.707107</td>
      <td>0.707107</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.196020</td>
      <td>-0.537894</td>
      <td>1</td>
      <td>-1.0</td>
      <td>-0.707107</td>
      <td>2.236068</td>
      <td>-0.707107</td>
      <td>0.707107</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.099668</td>
      <td>-0.864257</td>
      <td>1</td>
      <td>1.0</td>
      <td>-0.707107</td>
      <td>-0.447214</td>
      <td>-0.707107</td>
      <td>0.707107</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.797347</td>
      <td>-0.682945</td>
      <td>1</td>
      <td>-1.0</td>
      <td>1.414214</td>
      <td>-0.447214</td>
      <td>1.414214</td>
      <td>-1.414214</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.694362</td>
      <td>2.036746</td>
      <td>0</td>
      <td>1.0</td>
      <td>-0.707107</td>
      <td>-0.447214</td>
      <td>-0.707107</td>
      <td>0.707107</td>
    </tr>
  </tbody>
</table>
</div>



## 4. Feature Binning

Feature binning is the process that converts a numerical (either continuous and discrete) feature into a categorical feature represented by a set of ranges, or bins.

For example, instead of representing age as a single real-valued feature, we chop ranges of age into 3 discrete bins:
$$
\begin{equation*}
    young \in [ages \ 25  - 34],  \qquad
    middle \in [ages \ 35  - 44], \qquad
    old \in [ages \ 45  - 54]
\end{equation*}
$$

### 4.1. General Approach
To implement feature binning or discretization of the "Age" variable into categorical bins using Python, you can use the `pandas` library which provides a straightforward method called cut for binning continuous variables. Here's how you can convert the age into three categories based on the provided ranges:


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Income (€)</th>
      <th>Vehicle</th>
      <th>Kids</th>
      <th>Residence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>32</td>
      <td>95000</td>
      <td>none</td>
      <td>no</td>
      <td>downtown</td>
    </tr>
    <tr>
      <th>1</th>
      <td>46</td>
      <td>210000</td>
      <td>car</td>
      <td>yes</td>
      <td>downtown</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25</td>
      <td>75000</td>
      <td>truck</td>
      <td>yes</td>
      <td>suburbs</td>
    </tr>
    <tr>
      <th>3</th>
      <td>36</td>
      <td>30000</td>
      <td>car</td>
      <td>yes</td>
      <td>suburbs</td>
    </tr>
    <tr>
      <th>4</th>
      <td>29</td>
      <td>55000</td>
      <td>none</td>
      <td>no</td>
      <td>suburbs</td>
    </tr>
    <tr>
      <th>5</th>
      <td>54</td>
      <td>430000</td>
      <td>car</td>
      <td>yes</td>
      <td>downtown</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Define bins and their labels
bins = [24, 34, 44, 54]  # Extend ranges to include all possible ages in each group
labels = ['Young', 'Middle', 'Old']

# Perform binning
df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True)  # right=True means inclusive on the right
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Income (€)</th>
      <th>Vehicle</th>
      <th>Kids</th>
      <th>Residence</th>
      <th>Age Group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>32</td>
      <td>95000</td>
      <td>none</td>
      <td>no</td>
      <td>downtown</td>
      <td>Young</td>
    </tr>
    <tr>
      <th>1</th>
      <td>46</td>
      <td>210000</td>
      <td>car</td>
      <td>yes</td>
      <td>downtown</td>
      <td>Old</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25</td>
      <td>75000</td>
      <td>truck</td>
      <td>yes</td>
      <td>suburbs</td>
      <td>Young</td>
    </tr>
    <tr>
      <th>3</th>
      <td>36</td>
      <td>30000</td>
      <td>car</td>
      <td>yes</td>
      <td>suburbs</td>
      <td>Middle</td>
    </tr>
    <tr>
      <th>4</th>
      <td>29</td>
      <td>55000</td>
      <td>none</td>
      <td>no</td>
      <td>suburbs</td>
      <td>Young</td>
    </tr>
    <tr>
      <th>5</th>
      <td>54</td>
      <td>430000</td>
      <td>car</td>
      <td>yes</td>
      <td>downtown</td>
      <td>Old</td>
    </tr>
  </tbody>
</table>
</div>



This approach allows for clear and meaningful categorization of ages, which can be very useful for analysis or as a feature in machine learning models where age is an important factor.

### 4.2. Equal Width Binning

Equal width binning divides the range of values of a feature into bins with equal width.

Usually, we specify the number of bins as a hyper-parameter $K$, and then compute the width of each bin as

$$
\begin{equation}
    w = \Big[\frac{max^{(j)} - min^{(j)}}{K}\Big]
\end{equation}
$$
where 
$max^{(j)}$ and $min^{(j)}$ are the $j^{th}$ feature's maximum and minimum values, respectively.


The ranges of the $K$ bins are then
$$
\begin{equation}
\begin{split}
    Bin \ 1&: [min, \ min + w - 1] \\
    Bin \ 2&: [min+w, \ min + 2\cdot w - 1] \\
    ... \\
    Bin \ K&: [min + (K-1)\cdot w, \ max]
    \label{eq:equal_width_binning}
\end{split}
\end{equation}
$$

As an example of equal width binning, consider splitting the Age feature in the Amsterdam demographics dataset into $K=3$ bins.

The bin's width is:
$$
\begin{equation*}
    w = \Big[\frac{max-min}{x}\Big] = \Big[\frac{54-25}{3}\Big] = 9.7 \approx 10
\end{equation*}
$$

which we rounded to the nearest integer because Age values are always integers (in this dataset).

To implement equal width binning in Python and calculate each bin's range for the "Age" feature of the Amsterdam demographics dataset using $K=3$ bins, we can use the `numpy` library to help with calculations and then use `pandas` for binning. Here's how you can perform this task:


```python
# Number of bins
K = 3

# Calculate the width of each bin
min_age = df['Age'].min()
max_age = df['Age'].max()
width = (max_age - min_age) // K
```


```python
# Define bins using calculated width
bins = np.linspace(min_age, max_age, num=K+1)

print(bins)
```

    [25.         34.66666667 44.33333333 54.        ]



```python
# Create bin labels
labels = [f'Bin {i+1}' for i in range(K)]

print(labels)
```

    ['Bin 1', 'Bin 2', 'Bin 3']



```python
# Perform binning
df['Age Bin'] = pd.cut(df['Age'], bins=bins, labels=labels, include_lowest=True, right=True)
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Income (€)</th>
      <th>Vehicle</th>
      <th>Kids</th>
      <th>Residence</th>
      <th>Age Bin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>32</td>
      <td>95000</td>
      <td>none</td>
      <td>no</td>
      <td>downtown</td>
      <td>Bin 1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>46</td>
      <td>210000</td>
      <td>car</td>
      <td>yes</td>
      <td>downtown</td>
      <td>Bin 3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25</td>
      <td>75000</td>
      <td>truck</td>
      <td>yes</td>
      <td>suburbs</td>
      <td>Bin 1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>36</td>
      <td>30000</td>
      <td>car</td>
      <td>yes</td>
      <td>suburbs</td>
      <td>Bin 2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>29</td>
      <td>55000</td>
      <td>none</td>
      <td>no</td>
      <td>suburbs</td>
      <td>Bin 1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>54</td>
      <td>430000</td>
      <td>car</td>
      <td>yes</td>
      <td>downtown</td>
      <td>Bin 3</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
