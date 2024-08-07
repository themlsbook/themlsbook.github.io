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

(chapter12_part1)=

# Hyper-parameters Tuning

- This is a supplement material for the [Machine Learning Simplified](https://themlsbook.com) book. It sheds light on Python implementations of the topics discussed while all detailed explanations can be found in the book. 
- I also assume you know Python syntax and how it works. If you don't, I highly recommend you to take a break and get introduced to the language before going forward with my code. 
- This material can be downloaded as a Jupyter notebook (Download button in the upper-right corner -> `.ipynb`) to reproduce the code and play around with it. 


This notebook is a supplement for *Chapter 12. Model Tuning and Selection* of **Machine Learning For Everyone** book.

## 1. Required Libraries

This block imports all necessary libraries. numpy is used for array manipulations, sklearn provides tools for data mining and data analysis, and skopt is used for Bayesian optimization.


```{code-cell} ipython3
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical, Real
```

## 2. Generate Synthetic Data

Here, we generate a synthetic dataset with 1000 samples and 20 features, split into training and test sets. This dataset will be used to train and evaluate our Decision Tree models.


```{code-cell} ipython3
# Generate a binary classification dataset.
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## 3. Hyperparameter Tuning with GridSearchCV


GridSearchCV exhaustively searches through the defined parameter grid, evaluating model performance for each combination using cross-validation. The best parameters and their corresponding performance are then displayed.


```{code-cell} ipython3
# Define the parameter grid
param_grid = {
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 6]
}

# Initialize the GridSearchCV object
grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42), param_grid=param_grid, cv=5, verbose=1, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters and best score
print("Best parameters found: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)

# Evaluate on the test set
y_pred = grid_search.predict(X_test)
print("Accuracy on test set: ", accuracy_score(y_test, y_pred))
```

## 4. Hyperparameter Tuning with RandomizedSearchCV

RandomizedSearchCV offers a probabilistic approach, randomly selecting combinations from the parameter distribution. It's typically faster than GridSearchCV, especially when dealing with a large hyperparameter space or when every incremental improvement is not critical.


```{code-cell} ipython3
# Define the parameter distribution
param_dist = {
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': np.arange(2, 21),
    'min_samples_leaf': np.arange(1, 7)
}

# Initialize the RandomizedSearchCV object
random_search = RandomizedSearchCV(estimator=DecisionTreeClassifier(random_state=42), param_distributions=param_dist, n_iter=100, cv=5, verbose=1, random_state=42, scoring='accuracy')
random_search.fit(X_train, y_train)

# Best parameters and best score
print("Best parameters found: ", random_search.best_params_)
print("Best score: ", random_search.best_score_)

# Evaluate on the test set
y_pred = random_search.predict(X_test)
print("Accuracy on test set: ", accuracy_score(y_test, y_pred))
```

## 5. Hyperparameter Tuning with Bayesian Optimization


BayesSearchCV utilizes Bayesian optimization to search for optimal parameters. This method builds a probability model of the objective function and uses it to select the most promising parameters to evaluate in the true objective function.


```{code-cell} ipython3
# Define the parameter space
param_space = {
    'max_depth': Integer(10, 50),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 6)
}

# Initialize the BayesSearchCV object
bayes_search = BayesSearchCV(estimator=DecisionTreeClassifier(random_state=42), search_spaces=param_space, n_iter=32, cv=5, verbose=1, scoring='accuracy')
bayes_search.fit(X_train, y_train)

# Best parameters and best score
print("Best parameters found: ", bayes_search.best_params_)
print("Best score: ", bayes_search.best_score_)

# Evaluate on the test set
y_pred = bayes_search.predict(X_test)
print("Accuracy on test set: ", accuracy_score(y_test, y_pred))
```
