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

(chapter7_part3)=

# Data Augmentation

- This is a supplement material for the [Machine Learning Simplified](https://themlsbook.com) book. It sheds light on Python implementations of the topics discussed while all detailed explanations can be found in the book. 
- I also assume you know Python syntax and how it works. If you don't, I highly recommend you to take a break and get introduced to the language before going forward with my code. 
- This material can be downloaded as a Jupyter notebook (Download button in the upper-right corner -> `.ipynb`) to reproduce the code and play around with it. 


This notebook is a supplement for *Chapter 7. Data Preparation* of **Machine Learning For Everyone** book.

## 1. Required Libraries, Data & Variables

Let's import needed libraries:


```python
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
```

Imagine you obtained the following dataset from  Silver Suchs Bank. This dataset contains 55 observations of bank transaction over a certain period of time. The target column $Status$ has two classes: $Fraud$ for fraudulent transactions and $Legit$ for legal transactions. Imagine that out of 55 observations in the dataset, there are 50 legal transactions (class $Legit$) and only 5 fraudulent transactions (class $Fraud$). These two classes are imbalanced.


```python
# Data for 55 transactions, out of which 5 are Fraud class
data = {
    "#": range(1, 56),
    "date": [
        "21/08/2020", "24/12/2020", "10/04/2020", "13/03/2020", "08/10/2020", "02/04/2020",
        "15/05/2020", "18/07/2020", "20/06/2020", "22/08/2020", "27/11/2020", "30/01/2020",
        "14/02/2020", "17/04/2020", "19/06/2020", "21/08/2020", "26/12/2020", "29/02/2020",
        "12/03/2020", "15/05/2020", "17/07/2020", "19/09/2020", "23/10/2020", "25/12/2020",
        "28/02/2020", "10/01/2020", "13/03/2020", "15/05/2020", "17/07/2020", "19/09/2020",
        "22/11/2020", "24/01/2020", "27/03/2020", "29/05/2020", "31/07/2020", "02/10/2020",
        "04/12/2020", "06/02/2020", "09/04/2020", "11/06/2020", "13/08/2020", "16/10/2020",
        "18/12/2020", "20/02/2020", "23/04/2020", "25/06/2020", "27/08/2020", "30/10/2020",
        "02/12/2020", "04/02/2020", "07/04/2020", "09/06/2020", "11/08/2020", "14/10/2020",
        "16/12/2020"
    ],
    "time": [
        "02:00", "05:19", "18:06", "19:01", "15:34", "23:58",
        "00:45", "01:15", "02:30", "03:50", "04:20", "05:45",
        "06:55", "07:25", "08:15", "09:35", "10:10", "11:20",
        "12:05", "13:30", "14:50", "15:40", "16:30", "17:20",
        "18:00", "19:10", "20:05", "21:15", "22:50", "23:30",
        "00:25", "01:35", "02:45", "03:55", "04:50", "05:10",
        "06:25", "07:35", "08:45", "09:55", "10:50", "11:00",
        "12:15", "13:25", "14:35", "15:45", "16:40", "17:50",
        "18:05", "19:15", "20:25", "21:35", "22:45", "23:55",
        "00:05"
    ],
    "location": [
        "Amsterdam", "Dusseldorf", "Berlin", "Belgium", "Paris", "Amsterdam",
        "Dusseldorf", "Berlin", "Belgium", "Paris", "Amsterdam", "Dusseldorf",
        "Berlin", "Belgium", "Paris", "Amsterdam", "Dusseldorf", "Berlin",
        "Belgium", "Paris", "Amsterdam", "Dusseldorf", "Berlin", "Belgium",
        "Paris", "Amsterdam", "Dusseldorf", "Berlin", "Belgium", "Paris",
        "Amsterdam", "Dusseldorf", "Berlin", "Belgium", "Paris", "Amsterdam",
        "Dusseldorf", "Berlin", "Belgium", "Paris", "Amsterdam", "Dusseldorf",
        "Berlin", "Belgium", "Paris", "Amsterdam", "Dusseldorf", "Berlin",
        "Belgium", "Paris", "Amsterdam", "Dusseldorf", "Berlin", "Belgium",
        "Paris"
    ],
    "Status": [
        "Legit", "Fraud", "Legit", "Legit", "Legit", "Fraud",
        "Legit", "Legit", "Legit", "Legit", "Legit", "Legit",
        "Legit", "Legit", "Legit", "Legit", "Fraud", "Legit",
        "Legit", "Legit", "Legit", "Legit", "Legit", "Legit",
        "Legit", "Legit", "Legit", "Legit", "Legit", "Legit",
        "Legit", "Legit", "Legit", "Legit", "Legit", "Fraud",
        "Legit", "Legit", "Legit", "Legit", "Legit", "Legit",
        "Legit", "Legit", "Legit", "Legit", "Fraud", "Legit",
        "Legit", "Legit", "Legit", "Legit", "Legit", "Legit",
        "Legit"
    ]
}

# Create DataFrame
df_bank_transactions = pd.DataFrame(data)
```


```python
# Display the DataFrame
df_bank_transactions.head(10)
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
      <th>#</th>
      <th>date</th>
      <th>time</th>
      <th>location</th>
      <th>Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>21/08/2020</td>
      <td>02:00</td>
      <td>Amsterdam</td>
      <td>Legit</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>24/12/2020</td>
      <td>05:19</td>
      <td>Dusseldorf</td>
      <td>Fraud</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>10/04/2020</td>
      <td>18:06</td>
      <td>Berlin</td>
      <td>Legit</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>13/03/2020</td>
      <td>19:01</td>
      <td>Belgium</td>
      <td>Legit</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>08/10/2020</td>
      <td>15:34</td>
      <td>Paris</td>
      <td>Legit</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>02/04/2020</td>
      <td>23:58</td>
      <td>Amsterdam</td>
      <td>Fraud</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>15/05/2020</td>
      <td>00:45</td>
      <td>Dusseldorf</td>
      <td>Legit</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>18/07/2020</td>
      <td>01:15</td>
      <td>Berlin</td>
      <td>Legit</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>20/06/2020</td>
      <td>02:30</td>
      <td>Belgium</td>
      <td>Legit</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>22/08/2020</td>
      <td>03:50</td>
      <td>Paris</td>
      <td>Legit</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Calculate the number of 'Fraud' and 'Legit' observations
status_counts = df_bank_transactions['Status'].value_counts()

print(status_counts)
```

    Status
    Legit    50
    Fraud     5
    Name: count, dtype: int64


Fraud class is imbalanced as it contains only 5 observations. Imbalanced classes can create problems in ML classification if the difference between the minority and majority classes are significant. When we have a very few observations in one class and a lot of observations in another, we try to minimize the gap. 

One of the ways to do so is by using oversampling technique SMOTE.


```python
# Preprocess the data: Convert categorical variables to numeric
# Encoding 'location' and 'Status' for demonstration

le_location = LabelEncoder()
df_bank_transactions['location_encoded'] = le_location.fit_transform(df_bank_transactions['location'])

le_status = LabelEncoder()
df_bank_transactions['Status_encoded'] = le_status.fit_transform(df_bank_transactions['Status'])
```


```python
df_bank_transactions.head()
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
      <th>#</th>
      <th>date</th>
      <th>time</th>
      <th>location</th>
      <th>Status</th>
      <th>location_encoded</th>
      <th>Status_encoded</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>21/08/2020</td>
      <td>02:00</td>
      <td>Amsterdam</td>
      <td>Legit</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>24/12/2020</td>
      <td>05:19</td>
      <td>Dusseldorf</td>
      <td>Fraud</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>10/04/2020</td>
      <td>18:06</td>
      <td>Berlin</td>
      <td>Legit</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>13/03/2020</td>
      <td>19:01</td>
      <td>Belgium</td>
      <td>Legit</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>08/10/2020</td>
      <td>15:34</td>
      <td>Paris</td>
      <td>Legit</td>
      <td>4</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Define features and target variable
X = df_bank_transactions[['location_encoded']]  # Simplified feature set for demonstration
y = df_bank_transactions['Status_encoded']
```


```python
# Apply SMOTE to balance the dataset
smote = SMOTE(sampling_strategy={0: 50}, k_neighbors=4)  # Class label '0' corresponds to 'Fraud'
X_res, y_res = smote.fit_resample(X, y)
```


```python
# Check the new class distribution
print("New class distribution:", pd.Series(y_res).value_counts())
```

    New class distribution: Status_encoded
    1    50
    0    50
    Name: count, dtype: int64



```python
# Optionally, convert results back to a DataFrame and map encoded values back to original
resampled_data = pd.DataFrame(X_res, columns=['location_encoded'])
resampled_data['Status'] = le_status.inverse_transform(y_res)
```


```python
resampled_data.tail(10)
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
      <th>location_encoded</th>
      <th>Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>90</th>
      <td>0</td>
      <td>Fraud</td>
    </tr>
    <tr>
      <th>91</th>
      <td>3</td>
      <td>Fraud</td>
    </tr>
    <tr>
      <th>92</th>
      <td>0</td>
      <td>Fraud</td>
    </tr>
    <tr>
      <th>93</th>
      <td>3</td>
      <td>Fraud</td>
    </tr>
    <tr>
      <th>94</th>
      <td>0</td>
      <td>Fraud</td>
    </tr>
    <tr>
      <th>95</th>
      <td>3</td>
      <td>Fraud</td>
    </tr>
    <tr>
      <th>96</th>
      <td>2</td>
      <td>Fraud</td>
    </tr>
    <tr>
      <th>97</th>
      <td>2</td>
      <td>Fraud</td>
    </tr>
    <tr>
      <th>98</th>
      <td>0</td>
      <td>Fraud</td>
    </tr>
    <tr>
      <th>99</th>
      <td>1</td>
      <td>Fraud</td>
    </tr>
  </tbody>
</table>
</div>


