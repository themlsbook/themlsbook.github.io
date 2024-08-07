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

(chapter11_part1)=

# Maximum Margin Models

- This is a supplement material for the [Machine Learning Simplified](https://themlsbook.com) book. It sheds light on Python implementations of the topics discussed while all detailed explanations can be found in the book. 
- I also assume you know Python syntax and how it works. If you don't, I highly recommend you to take a break and get introduced to the language before going forward with my code. 
- This material can be downloaded as a Jupyter notebook (Download button in the upper-right corner -> `.ipynb`) to reproduce the code and play around with it. 


This notebook is a supplement for *Chapter 11. Maximum Margin Models* of **Machine Learning For Everyone** book.

## 1. Required Libraries

Let's import required libraries:


```{code-cell} ipython3
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
```

## 2. Create a Synthetic Dataset

To demonstrate the application of a Maximum Margin Model using both Linear SVM (Support Vector Machine) and Kernelized SVM, let's first create a synthetic dataset in Python. We will use this dataset to train and evaluate our models.

We'll use make_classification from scikit-learn to generate a binary classification dataset.


```{code-cell} ipython3
# Step 1: Create a synthetic dataset
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1, flip_y=0.1)

```


```{code-cell} ipython3
# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
```

## 2. Implement Linear SVM and Kernelized SVM

We will use the SVC (Support Vector Classifier) from scikit-learn, applying both linear and kernelized (e.g., RBF) approaches.


```{code-cell} ipython3
# Step 2: Implement SVM Models
# Linear SVM
linear_svm = SVC(kernel='linear', C=1.0)
linear_svm.fit(X_train, y_train)
```


```{code-cell} ipython3
# Kernelized SVM (RBF kernel)
rbf_svm = SVC(kernel='rbf', C=1.0, gamma='scale')
rbf_svm.fit(X_train, y_train)
```

## 3. Make Prediction


```{code-cell} ipython3
# Step 3: Evaluate the models
# Predictions from both models
y_pred_linear = linear_svm.predict(X_test)
y_pred_rbf = rbf_svm.predict(X_test)
```

## 4. Evaluate the Models


```{code-cell} ipython3
# Accuracy and Confusion Matrix
accuracy_linear = accuracy_score(y_test, y_pred_linear)
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
cm_linear = confusion_matrix(y_test, y_pred_linear)
cm_rbf = confusion_matrix(y_test, y_pred_rbf)
```


```{code-cell} ipython3
cm_linear
```


```{code-cell} ipython3
cm_rbf
```


```{code-cell} ipython3
# Print the results
print("Linear SVM Accuracy:", accuracy_linear)
print("Linear SVM Confusion Matrix:\n", cm_linear)
print("Kernelized SVM Accuracy:", accuracy_rbf)
print("Kernelized SVM Confusion Matrix:\n", cm_rbf)
```

## 5. Plotting the dataset and the decision boundary


```pyt{code-cell} ipython3hon
# Plotting the dataset and the decision boundary
def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=50, linewidth=1, facecolors='none', edgecolors='k')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

# Plotting
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='autumn')
plot_svc_decision_function(linear_svm)
plt.title("Linear SVM")

plt.subplot(1, 2, 2)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='autumn')
plot_svc_decision_function(rbf_svm)
plt.title("Kernelized SVM (RBF)")

plt.show()
```
