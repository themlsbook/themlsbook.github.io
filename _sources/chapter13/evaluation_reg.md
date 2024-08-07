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

(chapter13_part1)=

# Evaluation Metrics for Regression

- This is a supplement material for the [Machine Learning Simplified](https://themlsbook.com) book. It sheds light on Python implementations of the topics discussed while all detailed explanations can be found in the book. 
- I also assume you know Python syntax and how it works. If you don't, I highly recommend you to take a break and get introduced to the language before going forward with my code. 
- This material can be downloaded as a Jupyter notebook (Download button in the upper-right corner -> `.ipynb`) to reproduce the code and play around with it.


This notebook is a supplement for *Chapter 13. Model Evaluation* of **Machine Learning For Everyone** book.


## 1. Required Libraries

This block imports all necessary libraries. 


```{code-cell} ipython3
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
```

## 2. Univariate Regression

Generate synthetic data for Univariate Regression using `sklearn.datasets.make_regression`


```{code-cell} ipython3
# Univariate Regression: 1 feature
X_uni, y_uni = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)
X_train_uni, X_test_uni, y_train_uni, y_test_uni = train_test_split(X_uni, y_uni, test_size=0.2, random_state=42)
```

Next step is to train a regression model (`LinearRegression`)


```{code-cell} ipython3
# Univariate Model
model_uni = LinearRegression()
model_uni.fit(X_train_uni, y_train_uni)
```

Finally, to evaluate the model, we calculate the predicted values made by a model.


```{code-cell} ipython3
# Predictions
y_pred_uni = model_uni.predict(X_test_uni)
```

### 2.1. Mean Squated Error


```{code-cell} ipython3
mse_uni = mean_squared_error(y_test_uni, y_pred_uni)

print("Mean Squared Error (MSE):", mse_uni)
```

### 2.2. R-squared


```{code-cell} ipython3
r2_uni = r2_score(y_test_uni, y_pred_uni)

print("R-squared (R²):", r2_uni)
```

## 3. Multivariate Regression

Generate synthetic data for Univariate Regression using `sklearn.datasets.make_regression`


```{code-cell} ipython3
# Multivariate Regression: 3 features
X_multi, y_multi = make_regression(n_samples=100, n_features=3, noise=0.1, random_state=42)
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)
```

Next step is to train a regression model (`LinearRegression`)


```{code-cell} ipython3
# Multivariate Model
model_multi = LinearRegression()
model_multi.fit(X_train_multi, y_train_multi)
```

Finally, to evaluate the model, we calculate the predicted values made by a model.


```{code-cell} ipython3
# Predictions
y_pred_multi = model_multi.predict(X_test_multi)
```

### 3.1. MSE and R-squared


```{code-cell} ipython3
# MSE and R-squared
mse_multi = mean_squared_error(y_test_multi, y_pred_multi)
r2_multi = r2_score(y_test_multi, y_pred_multi)

print("Mean Squared Error (MSE):", mse_multi)
print("R-squared (R²):", r2_multi)
```

### 3.2. Adjusted R-squared


```{code-cell} ipython3
# Adjusted R-squared
n = len(y_test_multi)  # number of data points
p = X_test_multi.shape[1]  # number of predictors
adj_r2_multi = 1 - (1 - r2_multi) * ((n - 1) / (n - p - 1))
```


```{code-cell} ipython3
print("Adjusted R-squared:", adj_r2_multi)
```
