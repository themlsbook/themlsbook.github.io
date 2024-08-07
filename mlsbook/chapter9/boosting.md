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

(chapter9_part2)=


## Boosting Models

- This is a supplement material for the [Machine Learning Simplified](https://themlsbook.com) book. It sheds light on Python implementations of the topics discussed while all detailed explanations can be found in the book. 
- I also assume you know Python syntax and how it works. If you don't, I highly recommend you to take a break and get introduced to the language before going forward with my code. 
- This material can be downloaded as a Jupyter notebook (Download button in the upper-right corner -> `.ipynb`) to reproduce the code and play around with it. 


This notebook is a supplement for *Chapter 9. Ensemble Models* of **Machine Learning For Everyone** book.

## 1. Required Libraries, Data & Variables

Let's import the data and have a look at it:


```{code-cell} ipython3
import pandas as pd

data = {
    'Day': list(range(1, 31)),
    'Temperature': [
        'Cold', 'Hot', 'Cold', 'Hot', 'Hot',
        'Cold', 'Hot', 'Cold', 'Hot', 'Cold',
        'Hot', 'Cold', 'Hot', 'Cold', 'Hot',
        'Cold', 'Hot', 'Cold', 'Hot', 'Cold',
        'Hot', 'Cold', 'Hot', 'Cold', 'Hot',
        'Cold', 'Hot', 'Cold', 'Hot', 'Cold'
    ],
    'Humidity': [
        'Normal', 'Normal', 'Normal', 'High', 'High',
        'Normal', 'High', 'Normal', 'High', 'Normal',
        'High', 'Normal', 'High', 'Normal', 'High',
        'Normal', 'High', 'Normal', 'High', 'Normal',
        'High', 'Normal', 'High', 'Normal', 'High',
        'Normal', 'High', 'Normal', 'High', 'Normal'
    ],
    'Outlook': [
        'Rain', 'Rain', 'Sunny', 'Sunny', 'Rain',
        'Sunny', 'Rain', 'Sunny', 'Rain', 'Sunny',
        'Rain', 'Sunny', 'Rain', 'Sunny', 'Rain',
        'Sunny', 'Rain', 'Sunny', 'Rain', 'Sunny',
        'Rain', 'Sunny', 'Rain', 'Sunny', 'Rain',
        'Sunny', 'Rain', 'Sunny', 'Rain', 'Sunny'
    ],
    'Wind': [
        'Strong', 'Weak', 'Weak', 'Weak', 'Weak',
        'Strong', 'Weak', 'Weak', 'Weak', 'Strong',
        'Weak', 'Weak', 'Strong', 'Weak', 'Weak',
        'Weak', 'Strong', 'Weak', 'Weak', 'Weak',
        'Strong', 'Weak', 'Weak', 'Weak', 'Weak',
        'Strong', 'Weak', 'Weak', 'Weak', 'Strong'
    ],
    'Golf Played': [
        'No', 'No', 'Yes', 'Yes', 'Yes',
        'No', 'Yes', 'No', 'Yes', 'Yes',
        'No', 'Yes', 'No', 'Yes', 'Yes',
        'No', 'Yes', 'No', 'Yes', 'Yes',
        'No', 'Yes', 'No', 'Yes', 'Yes',
        'No', 'Yes', 'No', 'Yes', 'Yes'
    ]
}

# Converting the dictionary into a DataFrame
df = pd.DataFrame(data)
```


```{code-cell} ipython3
# Displaying the DataFrame
df.head(10)
```

## 2. Preparation of the Dataset

One-hot encoding the categorical variables


```{code-cell} ipython3
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False)
encoded_features = encoder.fit_transform(df[['Temperature', 'Humidity', 'Outlook', 'Wind']])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['Temperature', 'Humidity', 'Outlook', 'Wind']))
```

Visualizing the first 10 records of the encoded dataframe:


```{code-cell} ipython3
encoded_df.head(10)
```

Adding the encoded features back to the dataframe


```{code-cell} ipython3
df = df.join(encoded_df)

df.head(5)
```

Preparing the features by removing categorical variables.


```{code-cell} ipython3
X = df.drop(['Day', 'Temperature', 'Humidity', 'Outlook', 'Wind', 'Golf Played'], axis=1)
X.head(5)
```

Defining y:


```{code-cell} ipython3
y = df['Golf Played']

y
```

Splitting the dataset into training and testing sets


```{code-cell} ipython3
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## 3. Boosting Ensemble

### 3.1. Building a Boosting Ensemble

Creating the Gradient Boosting classifier


```{code-cell} ipython3
from sklearn.ensemble import GradientBoostingClassifier
```


```{code-cell} ipython3
# Creating the Gradient Boosting classifier
model = GradientBoostingClassifier(n_estimators=5, 
                                   learning_rate=0.1, 
                                   max_depth=3, 
                                   random_state=42)
model.fit(X_train, y_train)
```

### 3.2. Visualizing boosted ensemble


```{code-cell} ipython3
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
```


```{code-cell} ipython3
# Building 5 decision trees
feature_names = encoder.get_feature_names_out(['Temperature', 'Humidity', 'Outlook', 'Wind'])
trees = [DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42 + i) for i in range(5)]
for tree in trees:
    tree.fit(X_train, y_train)

# Plotting all 5 trees
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 4), dpi=300)
for i, tree in enumerate(trees):
    plot_tree(tree, feature_names=feature_names, class_names=['No', 'Yes'], filled=True, ax=axes[i])
    axes[i].set_title(f'Tree {i+1}')

plt.tight_layout()
plt.show()
```

### 3.3. Feature Importance


```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
```


```{code-cell} ipython3
# Feature importance
feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

# Plotting Feature Importance
plt.figure(figsize=(12, 6))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, np.array(X.columns)[sorted_idx])
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()

```

### 3.4. Predicting the Results

Predicting the test set results


```{code-cell} ipython3
y_pred = model.predict(X_test)
```


```{code-cell} ipython3
y_pred
```

### 3.5. Evaluating the model


```{code-cell} ipython3
from sklearn.metrics import accuracy_score, classification_report
```


```{code-cell} ipython3
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
```

## 4. Adaboost Algorithm

To implement the AdaBoost algorithm using the same dataset and scikit-learn library, we'll use the `AdaBoostClassifier`. AdaBoost (Adaptive Boosting) works by combining multiple weak classifiers into a single strong classifier. Each subsequent classifier focuses more on the samples that were misclassified by the previous ones, improving the ensemble's overall accuracy.

### 4.1. Building a Boosting Ensemble

Creating the Random Forest Classifier


```{code-cell} ipython3
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
```


```{code-cell} ipython3
# Creating and training the AdaBoost Classifier
# Using a DecisionTreeClassifier as the base classifier

base_estimator = DecisionTreeClassifier(max_depth=1)  # a stump (tree with depth 1)
adaboost = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=50, random_state=42)
adaboost.fit(X_train, y_train)
```

### 4.2. Predicting the Results


```{code-cell} ipython3
# Making predictions on the test set
y_pred = adaboost.predict(X_test)

y_pred
```

### 4.3. Evaluating the model


```{code-cell} ipython3
# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
```
