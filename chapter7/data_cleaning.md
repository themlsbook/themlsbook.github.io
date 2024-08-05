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

(chapter7_part1)=

# Data Cleaning

- This is a supplement material for the [Machine Learning Simplified](https://themlsbook.com) book. It sheds light on Python implementations of the topics discussed while all detailed explanations can be found in the book. 
- I also assume you know Python syntax and how it works. If you don't, I highly recommend you to take a break and get introduced to the language before going forward with my code. 
- This material can be downloaded as a Jupyter notebook (Download button in the upper-right corner -> `.ipynb`) to reproduce the code and play around with it. 


This notebook is a supplement for *Chapter 7. Data Preparation* of **Machine Learning For Everyone** book.

## 1. Required Libraries, Data & Variables

Let's import the data and have a look at it:


```python
import pandas as pd

# Define the data as a dictionary
data = {
    "Customer ID": [383, 1997, 698, 1314, 1314, 333, 1996],
    "State": ["Pennsylvania", "Californai", "California", "Iowa", "Iowa", "New York", "Washington"],
    "City": ["Drexel Hill", "Sacramento", "Los Angeles", "Fort Dodge", "Fort Dodge", "Brooklyn", None],
    "Postal Code": [19026, 94229, 90058, 50501, 50501, 11249, 98101],
    "Ship Date": ["23/08/2020", "07/03/2020", "14/09/2020", "29/02/2020", "29/02/2020", "14-09-2020", "19/05/2020"],
    "Purchase ($)": [190, 243, None, 193, 193, 298, 1]
}

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)

# Print the DataFrame to the console
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
      <th>Customer ID</th>
      <th>State</th>
      <th>City</th>
      <th>Postal Code</th>
      <th>Ship Date</th>
      <th>Purchase ($)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>383</td>
      <td>Pennsylvania</td>
      <td>Drexel Hill</td>
      <td>19026</td>
      <td>23/08/2020</td>
      <td>190.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1997</td>
      <td>Californai</td>
      <td>Sacramento</td>
      <td>94229</td>
      <td>07/03/2020</td>
      <td>243.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>698</td>
      <td>California</td>
      <td>Los Angeles</td>
      <td>90058</td>
      <td>14/09/2020</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1314</td>
      <td>Iowa</td>
      <td>Fort Dodge</td>
      <td>50501</td>
      <td>29/02/2020</td>
      <td>193.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1314</td>
      <td>Iowa</td>
      <td>Fort Dodge</td>
      <td>50501</td>
      <td>29/02/2020</td>
      <td>193.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>333</td>
      <td>New York</td>
      <td>Brooklyn</td>
      <td>11249</td>
      <td>14-09-2020</td>
      <td>298.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1996</td>
      <td>Washington</td>
      <td>None</td>
      <td>98101</td>
      <td>19/05/2020</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



Table above contains a hypothetical dirty dataset of online product orders. This dataset has a number of issues, such as incorrect data, missing data, duplicated data, irrelevant data, and improperly formatted data, that make it impossible to apply ML algorithms right away. This section discusses methods that can be used to clean this data set such that ML algorithms can be applied to it

## 2. Data Cleaning

### 2.1. Incorrect Data

Datasets may contain data that is clearly incorrect, such as spelling or syntax errors. The data point in the second row of Table has value “Californai” for its state feature, which is clearly a misspelling of the state “California”. If this mistake were left uncorrected, any ML algorithm built on this dataset would treat the two strings “Californai” and “California” differently. 

How can we identify incorrect data? One way to check whether a particular column has misspelled values is to look at its set of unique values, which is often much smaller than the set of all values itself.


```python
df.State.unique()
```




    array(['Pennsylvania', 'Californai', 'California', 'Iowa', 'New York',
           'Washington'], dtype=object)



We can fix misspelled 'Californai' with the code:


```python
df['State'] = df['State'].replace('Californai', 'California')
```

Revisiting the dataframe, the problem has been fixed.


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
      <th>Customer ID</th>
      <th>State</th>
      <th>City</th>
      <th>Postal Code</th>
      <th>Ship Date</th>
      <th>Purchase ($)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>383</td>
      <td>Pennsylvania</td>
      <td>Drexel Hill</td>
      <td>19026</td>
      <td>23/08/2020</td>
      <td>190.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1997</td>
      <td>California</td>
      <td>Sacramento</td>
      <td>94229</td>
      <td>07/03/2020</td>
      <td>243.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>698</td>
      <td>California</td>
      <td>Los Angeles</td>
      <td>90058</td>
      <td>14/09/2020</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1314</td>
      <td>Iowa</td>
      <td>Fort Dodge</td>
      <td>50501</td>
      <td>29/02/2020</td>
      <td>193.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1314</td>
      <td>Iowa</td>
      <td>Fort Dodge</td>
      <td>50501</td>
      <td>29/02/2020</td>
      <td>193.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>333</td>
      <td>New York</td>
      <td>Brooklyn</td>
      <td>11249</td>
      <td>14-09-2020</td>
      <td>298.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1996</td>
      <td>Washington</td>
      <td>None</td>
      <td>98101</td>
      <td>19/05/2020</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



### 2.2. Improperly Formatted Data

In some cases, we might have improperly formatted values. For instance, the Ship Date column in Table includes dates are improperly formatted, leading to misaligned date format. We need to standardize the format for all the dates, since an algorithm would treat the date 19-05-2020 and the date 19/05/2020 as two different dates, even though they are the same date in different formats.

We can fix these inconsistences by using `pd.to_datetime` with `format='mixed'` argument to handle a mixture of date formats dynamically. This can be particularly useful if the dates are not consistently in one format.


```python
# Convert 'Ship Date' to datetime format (ISO format by default)
df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='mixed')
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
      <th>Customer ID</th>
      <th>State</th>
      <th>City</th>
      <th>Postal Code</th>
      <th>Ship Date</th>
      <th>Purchase ($)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>383</td>
      <td>Pennsylvania</td>
      <td>Drexel Hill</td>
      <td>19026</td>
      <td>2020-08-23</td>
      <td>190.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1997</td>
      <td>California</td>
      <td>Sacramento</td>
      <td>94229</td>
      <td>2020-07-03</td>
      <td>243.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>698</td>
      <td>California</td>
      <td>Los Angeles</td>
      <td>90058</td>
      <td>2020-09-14</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1314</td>
      <td>Iowa</td>
      <td>Fort Dodge</td>
      <td>50501</td>
      <td>2020-02-29</td>
      <td>193.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1314</td>
      <td>Iowa</td>
      <td>Fort Dodge</td>
      <td>50501</td>
      <td>2020-02-29</td>
      <td>193.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>333</td>
      <td>New York</td>
      <td>Brooklyn</td>
      <td>11249</td>
      <td>2020-09-14</td>
      <td>298.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1996</td>
      <td>Washington</td>
      <td>None</td>
      <td>98101</td>
      <td>2020-05-19</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



Let's format the 'Ship Date' to "dd/mm/yyyy" after conversion


```python
df['Ship Date'] = df['Ship Date'].dt.strftime('%d/%m/%Y')

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
      <th>Customer ID</th>
      <th>State</th>
      <th>City</th>
      <th>Postal Code</th>
      <th>Ship Date</th>
      <th>Purchase ($)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>383</td>
      <td>Pennsylvania</td>
      <td>Drexel Hill</td>
      <td>19026</td>
      <td>23/08/2020</td>
      <td>190.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1997</td>
      <td>California</td>
      <td>Sacramento</td>
      <td>94229</td>
      <td>03/07/2020</td>
      <td>243.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>698</td>
      <td>California</td>
      <td>Los Angeles</td>
      <td>90058</td>
      <td>14/09/2020</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1314</td>
      <td>Iowa</td>
      <td>Fort Dodge</td>
      <td>50501</td>
      <td>29/02/2020</td>
      <td>193.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1314</td>
      <td>Iowa</td>
      <td>Fort Dodge</td>
      <td>50501</td>
      <td>29/02/2020</td>
      <td>193.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>333</td>
      <td>New York</td>
      <td>Brooklyn</td>
      <td>11249</td>
      <td>14/09/2020</td>
      <td>298.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1996</td>
      <td>Washington</td>
      <td>None</td>
      <td>98101</td>
      <td>19/05/2020</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



### 2.3. Duplicated Data

Duplicated data is another common problem that arises in practice. For example, provided dataset has duplicate observations in rows three and four, and in rows four and five. Duplicate data effectively doubles the weight that an ML algorithm gives to the data point and has the effect of incorrectly prioritizing some data points over others which can lead to a poor model. 

One of the ways to fix this is to use `drop_duplicates()` method in pandas:


```python
df_cleaned = df.drop_duplicates()
```


```python
df_cleaned
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
      <th>Customer ID</th>
      <th>State</th>
      <th>City</th>
      <th>Postal Code</th>
      <th>Ship Date</th>
      <th>Purchase ($)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>383</td>
      <td>Pennsylvania</td>
      <td>Drexel Hill</td>
      <td>19026</td>
      <td>23/08/2020</td>
      <td>190.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1997</td>
      <td>California</td>
      <td>Sacramento</td>
      <td>94229</td>
      <td>03/07/2020</td>
      <td>243.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>698</td>
      <td>California</td>
      <td>Los Angeles</td>
      <td>90058</td>
      <td>14/09/2020</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1314</td>
      <td>Iowa</td>
      <td>Fort Dodge</td>
      <td>50501</td>
      <td>29/02/2020</td>
      <td>193.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>333</td>
      <td>New York</td>
      <td>Brooklyn</td>
      <td>11249</td>
      <td>14/09/2020</td>
      <td>298.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1996</td>
      <td>Washington</td>
      <td>None</td>
      <td>98101</td>
      <td>19/05/2020</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



### 2.4. Missing Data

Missing data arises for a variety of reasons. For example, if the data is entered by a human being, he may have forgotten to input one or more values. Alternatively, data may be missing because it is genuinely unknown or unmeasured, such as, for example, a set of survey questions that were answered by some, but not all, customers. A missing value occurs in our running example for the purchase column in row three (Purchase column) and six (City column).

#### 2.4.1. Missing Value in Purchase Column
A missing value in Purchase column cannot be determined exactly and we need to make an educated guess at its value. For example, to impute a missing product order, we take the median order total.


```python
# Calculate the median of the 'Purchase ($)' column, excluding NaN values
median_purchase = df_cleaned['Purchase ($)'].median()


# Impute missing values in the 'Purchase ($)' column with the median
df_cleaned['Purchase ($)'].fillna(median_purchase, inplace=True)
```

    /var/folders/5y/7zvhsc3x5nx162713kvx9c1m0000gn/T/ipykernel_95810/3282763330.py:6: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
    The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.
    
    For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.
    
    
      df_cleaned['Purchase ($)'].fillna(median_purchase, inplace=True)
    /var/folders/5y/7zvhsc3x5nx162713kvx9c1m0000gn/T/ipykernel_95810/3282763330.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_cleaned['Purchase ($)'].fillna(median_purchase, inplace=True)



```python
df_cleaned
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
      <th>Customer ID</th>
      <th>State</th>
      <th>City</th>
      <th>Postal Code</th>
      <th>Ship Date</th>
      <th>Purchase ($)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>383</td>
      <td>Pennsylvania</td>
      <td>Drexel Hill</td>
      <td>19026</td>
      <td>23/08/2020</td>
      <td>190.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1997</td>
      <td>California</td>
      <td>Sacramento</td>
      <td>94229</td>
      <td>03/07/2020</td>
      <td>243.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>698</td>
      <td>California</td>
      <td>Los Angeles</td>
      <td>90058</td>
      <td>14/09/2020</td>
      <td>193.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1314</td>
      <td>Iowa</td>
      <td>Fort Dodge</td>
      <td>50501</td>
      <td>29/02/2020</td>
      <td>193.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>333</td>
      <td>New York</td>
      <td>Brooklyn</td>
      <td>11249</td>
      <td>14/09/2020</td>
      <td>298.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1996</td>
      <td>Washington</td>
      <td>None</td>
      <td>98101</td>
      <td>19/05/2020</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



#### 2.4.2. Missing Value in City Column
Sometimes the missing value can be determined exactly. For example, if the US State of an order is missing, but we have its zip code, we can determine its state exactly (assuming we have another table which maps zip codes to states) and fill it into the table.

To fill in missing values for the 'City' column using the zip code when you have another table that maps zip codes to cities, you can use `pandas` library to merge these dataframes effectively. Here's how you can perform this operation:

- Create a mapping table: This table will map zip codes to their corresponding cities.
- Merge this mapping table with your main data table: This merge operation will use the zip code as a key.
- Update the 'City' column in the main table: If the 'City' is missing but can be found through the mapping table, update it accordingly.



```python
# Mapping table that relates Postal Codes to Cities
zip_to_city = {
    "Postal Code": [94229, 50501, 98101, 11249, 90058, 19026],
    "City": ["Sacramento", "Fort Dodge", "Seattle", "Brooklyn", "Los Angeles", "Drexel Hill"]
}

df_mapping = pd.DataFrame(zip_to_city)
```


```python
df_mapping
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
      <th>Postal Code</th>
      <th>City</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>94229</td>
      <td>Sacramento</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50501</td>
      <td>Fort Dodge</td>
    </tr>
    <tr>
      <th>2</th>
      <td>98101</td>
      <td>Seattle</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11249</td>
      <td>Brooklyn</td>
    </tr>
    <tr>
      <th>4</th>
      <td>90058</td>
      <td>Los Angeles</td>
    </tr>
    <tr>
      <th>5</th>
      <td>19026</td>
      <td>Drexel Hill</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Merge the main DataFrame with the mapping DataFrame on 'Postal Code'
df_merged = df_cleaned.merge(df_mapping, on="Postal Code", how="left", suffixes=('', '_mapped'))
```


```python
df_merged
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
      <th>Customer ID</th>
      <th>State</th>
      <th>City</th>
      <th>Postal Code</th>
      <th>Ship Date</th>
      <th>Purchase ($)</th>
      <th>City_mapped</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>383</td>
      <td>Pennsylvania</td>
      <td>Drexel Hill</td>
      <td>19026</td>
      <td>23/08/2020</td>
      <td>190.0</td>
      <td>Drexel Hill</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1997</td>
      <td>California</td>
      <td>Sacramento</td>
      <td>94229</td>
      <td>03/07/2020</td>
      <td>243.0</td>
      <td>Sacramento</td>
    </tr>
    <tr>
      <th>2</th>
      <td>698</td>
      <td>California</td>
      <td>Los Angeles</td>
      <td>90058</td>
      <td>14/09/2020</td>
      <td>193.0</td>
      <td>Los Angeles</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1314</td>
      <td>Iowa</td>
      <td>Fort Dodge</td>
      <td>50501</td>
      <td>29/02/2020</td>
      <td>193.0</td>
      <td>Fort Dodge</td>
    </tr>
    <tr>
      <th>4</th>
      <td>333</td>
      <td>New York</td>
      <td>Brooklyn</td>
      <td>11249</td>
      <td>14/09/2020</td>
      <td>298.0</td>
      <td>Brooklyn</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1996</td>
      <td>Washington</td>
      <td>NaN</td>
      <td>98101</td>
      <td>19/05/2020</td>
      <td>1.0</td>
      <td>Seattle</td>
    </tr>
  </tbody>
</table>
</div>



### 2.5. Outliers

An outlier is an observation that differs significantly from other observations. Outliers may be problematic for one of two reasons: 
- first, an outlier may simply not be representative of data that we will see at test time (in a new dataset); 
- second, many ML algorithms are sensitive to severe outliers and often learn models which focuses too heavily on outliers and consequently make poor predictions on the rest of the data points. 

There are no hard and fast rules about how to classify a point as an outlier and whether or not to remove it from the dataset. Usually, you will build ML models several times, both with and without outliers, and with different methods of outlier categorization. This subsection shows a statistical test that is commonly performed in practice to determine an outlier.

How can we use statistical metrics to determine if a data point is an outlier? The simplest way is to identify if a datapoint is too far away from the average value.


```python
mean_value = df_cleaned['Purchase ($)'].mean()
std_dev = df_cleaned['Purchase ($)'].std()

print(f"Mean = {mean_value}, Standard Deviation = {std_dev}")
```

    Mean = 186.33333333333334, Standard Deviation = 100.13124720419029


Suppose we set a range of acceptable values of k = 3 standard deviations. Then: 


```python
# Find outliers based on the defined threshold
k=3

lower_bound = mean_value - k * std_dev
upper_bound = mean_value + k * std_dev

print(f"Anything below {lower_bound} and above {upper_bound} is considered as an outlier.")
```

    Anything below -114.06040827923752 and above 486.7270749459042 is considered as an outlier.


But since we cannot have a purchase with a negative sum, outlier would be above 486.72. In this data set, there are no outliers present.
