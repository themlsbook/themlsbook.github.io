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

(chapter8_part2)=


## Decision Trees in Classification

- This is a supplement material for the [Machine Learning Simplified](https://themlsbook.com) book. It sheds light on Python implementations of the topics discussed while all detailed explanations can be found in the book. 
- I also assume you know Python syntax and how it works. If you don't, I highly recommend you to take a break and get introduced to the language before going forward with my code. 
- This material can be downloaded as a Jupyter notebook (Download button in the upper-right corner -> `.ipynb`) to reproduce the code and play around with it. 


This notebook is a supplement for *Chapter 8. Decision Trees* of **Machine Learning For Everyone** book.

## 1. Required Libraries, Data & Variables

Let's import the data and have a look at it:


```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
%config InlineBackend.figure_format = 'retina' #to make sharper and prettier plots

# Creating the DataFrame based on the provided data
data = {
    'x1': [0.25, 0.60, 0.71, 1.20, 1.75, 2.26, 2.50, 2.50, 2.88, 2.91],
    'x2': [1.41, 0.39, 1.29, 2.30, 0.59, 1.70, 1.35, 2.90, 0.61, 2.00],
    'Color': ['blue', 'blue', 'blue', 'blue', 'blue', 'green', 'green', 'green', 'green', 'green']
}

# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
df
```

## 2. Visualizing Dataframe


```{code-cell} ipython3
# Plotting
fig, ax = plt.subplots()
colors = {'blue': 'blue', 'green': 'green'}

# Group by color and then plot each group
for key, group in df.groupby('Color'):
    group.plot(ax=ax, kind='scatter', x='x1', y='x2', label=key, color=colors[key])

# Setting plot labels and title
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_title('Scatter Plot of Colors')

# Display the legend
ax.legend(title='Point Color')

# Show the plot
plt.show()
```

## 3. Preprocessing Dataframe


```{code-cell} ipython3
df['Color'] = df['Color'].map({'blue': 0, 'green': 1})
```


```{code-cell} ipython3
df
```

## 4. Training a Decision Tree with Gini


```{code-cell} ipython3
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```

#### 4.1. Splitting into X and y


```{code-cell} ipython3
X = df[['x1', 'x2']]
y = df['Color']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

#### 4.2. Building the Decision Tree Classifier


```{code-cell} ipython3
tree_classifier = DecisionTreeClassifier(criterion='gini', random_state=42)
tree_classifier.fit(X_train, y_train)
```

#### 4.3. Predict and Evaluate the Model


```{code-cell} ipython3
y_pred = tree_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the decision tree model:", accuracy)
```

#### 4.4. Visualize the Tree (optional)


```{code-cell} ipython3
from sklearn import tree
import graphviz

dot_data = tree.export_graphviz(tree_classifier, out_file=None, 
                                feature_names=['x1', 'x2'],  
                                class_names=['blue', 'green'],
                                filled=True, rounded=True,  
                                special_characters=True)  
graph = graphviz.Source(dot_data)  
```


```{code-cell} ipython3
# Visualize the decision tree
graph
```

## 5. Training a Decision Tree with Entropy

#### 4.1. Splitting into X and y


```{code-cell} ipython3
X = df[['x1', 'x2']]
y = df['Color']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

#### 4.2. Building the Decision Tree Classifier


```{code-cell} ipython3
tree_classifier = DecisionTreeClassifier(criterion='entropy', random_state=42)
tree_classifier.fit(X_train, y_train)
```

#### 4.3. Predict and Evaluate the Model


```{code-cell} ipython3
y_pred = tree_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the decision tree model:", accuracy)
```

#### 4.4. Visualize the Tree (optional)


```{code-cell} ipython3
from sklearn import tree
import graphviz

dot_data = tree.export_graphviz(tree_classifier, out_file=None, 
                                feature_names=['x1', 'x2'],  
                                class_names=['blue', 'green'],
                                filled=True, rounded=True,  
                                special_characters=True)  
graph = graphviz.Source(dot_data)  
```


```{code-cell} ipython3
# Visualize the decision tree
graph
```
