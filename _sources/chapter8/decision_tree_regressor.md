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

(chapter8_part1)=


## Decision Trees in Classification

- This is a supplement material for the [Machine Learning Simplified](https://themlsbook.com) book. It sheds light on Python implementations of the topics discussed while all detailed explanations can be found in the book. 
- I also assume you know Python syntax and how it works. If you don't, I highly recommend you to take a break and get introduced to the language before going forward with my code. 
- This material can be downloaded as a Jupyter notebook (Download button in the upper-right corner -> `.ipynb`) to reproduce the code and play around with it. 


This notebook is a supplement for *Chapter 7. Data Preparation* of **Machine Learning For Everyone** book.

## 1. Required Libraries, Data & Variables

Let's import the data and have a look at it:


```{code-cell} ipython3
import pandas as pd
%config InlineBackend.figure_format = 'retina' #to make sharper and prettier plots

# Data from the provided table
data = {
    'Day': list(range(1, 13)),
    'Temperature': ['Hot', 'Hot', 'Hot', 'Cold', 'Cold', 'Cold', 'Cold', 'Hot', 'Hot', 'Cold', 'Cold', 'Cold'],
    'Humidity': ['High', 'High', 'High', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'High', 'Normal'],
    'Outlook': ['Sunny', 'Sunny', 'Rain', 'Rain', 'Rain', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Rain', 'Rain', 'Sunny'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'Golf Hours Played': [25, 30, 42, 32, 23, 35, 38, 43, 48, 12, 24, 22]
}

# Create DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
df
```

## 2. Preprocessing Dataframe


```{code-cell} ipython3
from sklearn.preprocessing import LabelEncoder
```


```{code-cell} ipython3
# Encode categorical variables
label_encoder = LabelEncoder()
df['Temperature'] = label_encoder.fit_transform(df['Temperature'])
df['Humidity'] = label_encoder.fit_transform(df['Humidity'])
df['Outlook'] = label_encoder.fit_transform(df['Outlook'])
df['Wind'] = label_encoder.fit_transform(df['Wind'])
```

Let's look at our dataset after preprocessing:


```{code-cell} ipython3
df
```

## 3. Training a Decision Tree with SDR


```{code-cell} ipython3
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
```

#### 3.1. Splitting into X and y


```{code-cell} ipython3
# Features and target variable
X = df[['Temperature', 'Humidity', 'Outlook', 'Wind']]
y = df['Golf Hours Played']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

#### 3.2. Building the Decision Tree Regressor


```{code-cell} ipython3
# Standard deviation reduction isn't a direct option in scikit-learn, so we use the default "mse" which is mean squared error
tree_regressor = DecisionTreeRegressor(criterion='squared_error', random_state=42)
tree_regressor.fit(X_train, y_train)
```

#### 3.3. Predict and Evaluate the Model


```{code-cell} ipython3
from sklearn.metrics import mean_squared_error
```


```{code-cell} ipython3
y_pred = tree_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
```

#### 3.4. Visualize the Tree (optional)


```{code-cell} ipython3
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plot_tree(tree_regressor, filled=True, feature_names=['Temperature', 'Humidity', 'Outlook', 'Wind'])

# Visualize the decision tree
plt.show()
```
