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


```{code-cell} ipython3
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
```

Imagine you obtained the following dataset from  Silver Suchs Bank. This dataset contains 55 observations of bank transaction over a certain period of time. The target column $Status$ has two classes: $Fraud$ for fraudulent transactions and $Legit$ for legal transactions. Imagine that out of 55 observations in the dataset, there are 50 legal transactions (class $Legit$) and only 5 fraudulent transactions (class $Fraud$). These two classes are imbalanced.


```{code-cell} ipython3
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


```{code-cell} ipython3
# Display the DataFrame
df_bank_transactions.head(10)
```


```{code-cell} ipython3
# Calculate the number of 'Fraud' and 'Legit' observations
status_counts = df_bank_transactions['Status'].value_counts()

print(status_counts)
```

Fraud class is imbalanced as it contains only 5 observations. Imbalanced classes can create problems in ML classification if the difference between the minority and majority classes are significant. When we have a very few observations in one class and a lot of observations in another, we try to minimize the gap. 

One of the ways to do so is by using oversampling technique SMOTE.


```{code-cell} ipython3
# Preprocess the data: Convert categorical variables to numeric
# Encoding 'location' and 'Status' for demonstration

le_location = LabelEncoder()
df_bank_transactions['location_encoded'] = le_location.fit_transform(df_bank_transactions['location'])

le_status = LabelEncoder()
df_bank_transactions['Status_encoded'] = le_status.fit_transform(df_bank_transactions['Status'])
```


```{code-cell} ipython3
df_bank_transactions.head()
```


```{code-cell} ipython3
# Define features and target variable
X = df_bank_transactions[['location_encoded']]  # Simplified feature set for demonstration
y = df_bank_transactions['Status_encoded']
```


```{code-cell} ipython3
# Apply SMOTE to balance the dataset
smote = SMOTE(sampling_strategy={0: 50}, k_neighbors=4)  # Class label '0' corresponds to 'Fraud'
X_res, y_res = smote.fit_resample(X, y)
```


```{code-cell} ipython3
# Check the new class distribution
print("New class distribution:", pd.Series(y_res).value_counts())
```


```{code-cell} ipython3
# Optionally, convert results back to a DataFrame and map encoded values back to original
resampled_data = pd.DataFrame(X_res, columns=['location_encoded'])
resampled_data['Status'] = le_status.inverse_transform(y_res)
```


```{code-cell} ipython3
resampled_data.tail(10)
```
