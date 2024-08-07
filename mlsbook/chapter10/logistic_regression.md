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

(chapter10_part1)=



# Logistic Regression Classifier

- This is a supplement material for the [Machine Learning Simplified](https://themlsbook.com) book. It sheds light on Python implementations of the topics discussed while all detailed explanations can be found in the book. 
- I also assume you know Python syntax and how it works. If you don't, I highly recommend you to take a break and get introduced to the language before going forward with my code. 
- This material can be downloaded as a Jupyter notebook (Download button in the upper-right corner -> `.ipynb`) to reproduce the code and play around with it. 


This notebook is a supplement for *Chapter 10. Logit Models* of **Machine Learning For Everyone** book.

## 1. Required Libraries & Functions
Before we start, we need to import few libraries and functions that we will use in this jupyterbook. You don't need to understand what those functions do for now.


```python
import numpy as np
from sklearn import datasets
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import warnings
import math
import matplotlib.pyplot as plt 
import scipy.optimize as opt

warnings.filterwarnings('ignore')
```

## 2. Problem representation

Let's revisit *Chapter 10: Logit Models* from the book, where we examine a hypothetical dataset containing data on breast cancer. This dataset includes various parameters related to individual women and their corresponding outcomes—specifically, whether they have been diagnosed with breast cancer.


```python
data = datasets.load_breast_cancer()

pd.DataFrame(data['data'])
```

We aim to simplify the detection of this disease and classify whether a woman potentially has it or not. To achieve this, we can utilize Logistic Regression Classification. Next, we will explain how to implement this technique from scratch, ensuring that everyone can grasp its workings comprehensively.

## 3. Coding Logistic Regression From Scratch

To provide a structured overview of our upcoming tasks, let's outline an agenda for this section. We'll focus on coding four key components necessary for implementing our custom logistic regression model:

- The sigmoid function
- The intercept
- The cost function
- The solvers

We will discuss each component step-by-step, exploring the underlying concepts and demonstrating how to translate these ideas into Python code.

### 3.1 Sigmoid

We have previously covered the sigmoid function. Essentially, coding this function is a straightforward task that requires just a single line of code. To deepen your understanding of the underlying logic, you can use a slider to manipulate the value of 'z'—the output from linear regression. This interactive approach will help you visualize how changes in 'z' affect the sigmoid output.


```python
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider
import ipywidgets as widgets

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def plot_sigmoid(z):
    z_values = np.linspace(-10, 10, 400)
    sigmoid_values = sigmoid(z_values)

    plt.figure(figsize=(10, 6))
    plt.plot(z_values, sigmoid_values, label='Sigmoid function')
    plt.scatter([z], [sigmoid(z)], color='red', zorder=5)
    plt.axvline(z, color='red', linestyle='--', label=f'z = {z:.2f}')
    plt.axhline(sigmoid(z), color='red', linestyle='--', label=f'Sigmoid(z) = {sigmoid(z):.2f}')
    plt.title('Sigmoid Function')
    plt.xlabel('z')
    plt.ylabel('Sigmoid(z)')
    plt.legend()
    plt.grid(True)
    plt.ylim(-0.1, 1.1)
    plt.show()

z_slider = FloatSlider(min=-10, max=10, step=0.1, value=0, description='z')

interact(plot_sigmoid, z=z_slider)
```


```python
def sigmoid(z):
        return 1 / (1 + np.exp(-z))
```

### 3.2 Intercept

The intercept helps handle the bias that's naturally present in real data. Without the intercept, we'd assume that with no influence from our features, the chances of seeing a '0' or a '1' are equally split at 50-50. But in many cases, that's not quite right. In our problem, the ratio of women without any disease is much higher, so the model should take this into account.

From mathematical POV, we have a basic equation of Logistic Regression:
$$P(y=1|x) = sigmoid(w_0 + \sum^n_{i=1} w_i x_i) \text{, where}$$

$P(y=1|x)$ - probability of classifying our output as 1, 

$\sum^n_{i=1} w_i x_i$ - logit function and 

$w_0$ - intercept term.

To add the possibility to have bias, i.e. intercept, we add an extra feature that's always set to a constant value of 1 and added into our matrix. By doing this, the model has a place to store the value of intercept by multiplying it on 1.

$$
\begin{bmatrix}
    1 & x_{1,1} & x_{1,2} & \ldots & x_{1,n} \\
    1 & x_{2,1} & x_{2,2} & \ldots & x_{2,n} \\
    \vdots & \vdots & \vdots & & \vdots \\
    1 & x_{m,1} & x_{m,2} & \ldots & x_{m,n} \\
\end{bmatrix}
\begin{bmatrix}
    w_0 \\
    w_1 \\
    \vdots \\
    w_n
\end{bmatrix}
=
\begin{bmatrix}
    y_1 \\
    y_2 \\
    \vdots \\
    y_m
\end{bmatrix}
$$

In this matrix multiplication, $w_0$ will be the intercept value and multiplied by 1 in our feature matrix.

As can be seen, what we need to do is quite ease: add one more column to the data, so the code would be


```python
def add_intercept(X):
        intercept = np.ones((X.shape[0], 1))
        return np.hstack((intercept, X))
```

However, to better understand how it really affects the internal process, we can build a simple vizual, where we can set different intercept values and get the skewed sigmoid. As you could guess, the higher the intercept value, the more sigmoid skewed to the right (because that means that we need lower value of linear regression output 'z' as intercept will compesate it)


```python
def plot_sigmoid_with_intercept(intercept):
    z_values = np.linspace(-10, 10, 400)
    sigmoid_values_no_intercept = sigmoid(z_values)
    sigmoid_values_with_intercept = sigmoid(z_values + intercept)

    plt.figure(figsize=(10, 6))
    plt.plot(z_values, sigmoid_values_no_intercept, label='Sigmoid without Intercept')
    plt.plot(z_values, sigmoid_values_with_intercept, label=f'Sigmoid with Intercept {intercept}')
    plt.axhline(0.5, color='gray', linestyle='--', label='Decision Boundary (0.5)')
    plt.title('Effect of Intercept on Sigmoid Function')
    plt.xlabel('z')
    plt.ylabel('Sigmoid(z)')
    plt.legend()
    plt.grid(True)
    plt.ylim(-0.1, 1.1)
    plt.show()

intercept_slider = FloatSlider(min=-7.5, max=7.5, step=0.1, value=0, description='intercept')

interact(plot_sigmoid_with_intercept, intercept=intercept_slider)

```

### 3.3 Cost function

In the book, we've already explored the role of cost functions and the rationale for applying penalties to them. Implementing this in Python isn't challenging; it primarily involves configuring various parameters and outputting an equation.

In our case, the cost function consistently uses cross-entropy loss. All that's required is to define a set of predefined parameters to incorporate into the function.


```python
m = len(y) # Number of training examples
h = self.sigmoid(np.dot(X, w)) # Predicted probabilities for the positive class using the sigmoid function
epsilon = 1e-5 # Small constant to prevent log(0) situations, ensuring numerical stability

cost = -1/m * (np.dot(y, np.log(h + epsilon)) + np.dot((1 - y), np.log(1 - h + epsilon))) # Compute the cross-entropy loss
```

After this, we need to add some conditional operators which will define the regularization we use (if we use any). For each of this regularization we simply add its value to the defined cost function and get the final cost for the model


```python
# Regularizations
if penalty == 'l1':
    cost += (1/self.C) * np.sum(np.abs(w[1:]))
elif penalty == 'l2':
    cost += (1/(2*self.C)) * np.sum(w[1:]**2)
elif self.penalty == 'elasticnet':
    l1_penalty = self.l1_ratio * (1/self.C) * np.sum(np.abs(w[1:]))
    l2_penalty = (1 - self.l1_ratio) * (1/(2*self.C)) * np.sum(w[1:]**2)
    cost += l1_penalty + l2_penalty
elif penalty is not None and self.penalty != 'none': # Handler for both None's
    raise ValueError('Invalid penalty type')
```

Finally, if mix all together, we get


```python
def cost_function(self, w, X, y):
        m = len(y) # Number of training examples
        h = self.sigmoid(np.dot(X, w)) # Predicted probabilities for the positive class using the sigmoid function
        epsilon = 1e-5 # Small constant to prevent log(0) situations, ensuring numerical stability
        cost = -1/m * (np.dot(y, np.log(h + epsilon)) + np.dot((1 - y), np.log(1 - h + epsilon))) # Compute the cross-entropy loss

        # Regularizations
        if self.penalty == 'l1':
            cost += (1/self.C) * np.sum(np.abs(w[1:]))
        elif self.penalty == 'l2':
            cost += (1/(2*self.C)) * np.sum(w[1:]**2)
        elif self.penalty == 'elasticnet':
            l1_penalty = self.l1_ratio * (1/self.C) * np.sum(np.abs(w[1:]))
            l2_penalty = (1 - self.l1_ratio) * (1/(2*self.C)) * np.sum(w[1:]**2)
            cost += l1_penalty + l2_penalty
        elif self.penalty is not None and self.penalty != 'none': # Handler for both None's
            raise ValueError('Invalid penalty type')
        
        return cost
```

Also, if you haven't fully understood how elastic net works, here is a simple vizualization of it. By choosing l1 ration we define, will it be more like Lasso-like or Ridge-like


```python
def l1_cost(theta):
    return np.abs(theta)

def l2_cost(theta):
    return theta**2

def elasticnet_cost(theta, l1_ratio=0.5):
    return l1_ratio * l1_cost(theta) + (1 - l1_ratio) * l2_cost(theta)

def plot_cost_functions(l1_ratio):
    theta_values = np.linspace(-10, 10, 400)
    
    l1_values = l1_cost(theta_values)
    l2_values = l2_cost(theta_values)
    elasticnet_values = elasticnet_cost(theta_values, l1_ratio)

    plt.figure(figsize=(10, 6))
    plt.plot(theta_values, l1_values, label='L1 Cost (Lasso)', linestyle='--')
    plt.plot(theta_values, l2_values, label='L2 Cost (Ridge)', linestyle='-.')
    plt.plot(theta_values, elasticnet_values, label=f'ElasticNet Cost (l1_ratio={l1_ratio:.2f})')
    plt.title('Regularization Cost Functions')
    plt.xlabel('Model Parameter (θ)')
    plt.ylabel('Cost')
    plt.legend()
    plt.grid(True)
    plt.ylim(-0.1, 2)
    plt.xlim(-2, 2)
    plt.show()

l1_ratio_slider = FloatSlider(min=0, max=1, step=0.01, value=0.5, description='l1_ratio')

interact(plot_cost_functions, l1_ratio=l1_ratio_slider)
```

### 3.4 Solvers

Finally, the hardest thing to possibly understand are solvers. They were mainly not covered in the book, but this is mostly the most crucial part of the regression, because it does all the 'mathematical' part of predicting the unseen target. There are 7 solvers which will be covered here one-by-one

### 3.4.1 Gradient descent

The first solver is not presented in the sklearn library, however, it is a good starting point in understanding how they work in terms of code. The idea behind gradient is already explained in `add link to book`. 

If we recap some thing from there, the gradient descent is used as one of the bacis optimization algorithms to minimize the cost function. By iteratively updating parameters by moving in the opposite direction of the gradient of the cost function. This movement is proportional to the learning rate, which controls the step size of each update. We stop this movement when the changes are becoming too small.

Here we can see how it works: we start from random point on the somewhere and doing steps in the direction of the optimal point. Also, we do not reach it, because the next step is too small (as exlpained before).


```python
n_features = 2
n_objects = 300

np.random.seed(100)
w_true = np.random.normal(size=(n_features, ))

X = np.random.uniform(-5, 5, (n_objects, n_features))
X *= (np.arange(n_features) * 2 + 1)[np.newaxis, :]  # for different scales

Y = X.dot(w_true) + np.random.normal(0, 1, (n_objects))

w_0 = np.random.uniform(-1, 1, (n_features))

batch_size = 10
num_steps = 50

step_size = 1e-2

w = w_0.copy()
w_list = [w.copy()]

for i in range(num_steps):
    w -= 2 * step_size * np.dot(X.T, np.dot(X, w) - Y) / Y.shape[0]
    w_list.append(w.copy())

w_list = np.array(w_list)

import plotly.graph_objects as go


def compute_limits(w_list):
    dx = np.max(np.abs(w_list[:, 0] - w_true[0])) * 1.1
    dy = np.max(np.abs(w_list[:, 1] - w_true[1])) * 1.1
    
    return (w_true[0] - dx, w_true[0] + dx), (w_true[1] - dy, w_true[1] + dy)


def compute_levels(x_range, y_range, num: int = 100):
    x, y = np.linspace(x_range[0], x_range[1], num), np.linspace(y_range[0], y_range[1], num)
    A, B = np.meshgrid(x, y)

    levels = np.empty_like(A)

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            w_tmp = np.array([A[i, j], B[i, j]])
            levels[i, j] = np.mean(np.power(np.dot(X, w_tmp) - Y, 2))
            
    return x, y, levels


def make_contour(x, y, levels, name: str=None):
    return go.Contour(
        x=x,
        y=y,
        z=levels,
        contours_coloring='lines',
        line_smoothing=1,
        line_width=2,
        ncontours=100,
        opacity=0.5,
        name=name
    )


def make_arrow(figure, x, y):
    x, dx = x
    y, dy = y
    
    figure.add_annotation(
        x=x + dx,
        y=y + dy,
        ax=x,
        ay=y,
        xref='x',
        yref='y',
        text='',
        showarrow=True,
        axref='x',
        ayref='y',
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
    )


def plot_trajectory(w_list, name):
    # compute limits
    x_range, y_range = compute_limits(w_list)
    
    # compute level set
    x, y, levels = compute_levels(x_range, y_range)
    
    # plot levels
    contour = make_contour(x, y, levels, 'Loss function levels')

    # plot weights
    w_path = go.Scatter(
        x=w_list[:, 0][:-1],
        y=w_list[:, 1][:-1],
        mode='lines+markers',
        name='W',
        marker=dict(size=7, color='red')
    )

    # plot final weight
    w_final = go.Scatter(
        x=[w_list[:, 0][-1]],
        y=[w_list[:, 1][-1]],
        mode='markers',
        name='W_final',
        marker=dict(size=10, color='black'),
    )
    
    # plot true optimum    
    w_true_point = go.Scatter(
        x=[w_true[0]],
        y=[w_true[1]],
        mode='markers',
        name='W_true',
        marker=dict(size=10, color='black'),
        marker_symbol='star'
    )
    
    # make the figure
    fig = go.Figure(data=[contour, w_path, w_final, w_true_point])

    fig.update_xaxes(type='linear', range=x_range)
    fig.update_yaxes(type='linear', range=y_range)

    fig.update_layout(title=name)

    fig.update_layout(
        autosize=True,
#         width=700,
        margin=dict(
            l=50,
            r=50,
            b=50,
            t=100,
            pad=4
        ),
        paper_bgcolor='LightSteelBlue',
    )

    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))

    fig.update_traces(showlegend=True)

    fig.show()
    
plot_trajectory(w_list, 'Gradient descent')
```

In order to do it by ourself, firstly we need to define the function to calculate the gradient. To do this, we need to calculate the predicted probabilities for each class, which will tell us about how good we are at precising the true label. After this, we aggregate the errors weighted by features to get the gradient of the loss. Finally, we average it by dividing it on the number of observations to get the to scale it appropriately. 


```python
def gradient(self, w, X, y):
        m = len(y) # Number of training examples
        h = self.sigmoid(np.dot(X, w)) # Predicted probabilities for the positive class using the sigmoid function
        gradient = 1/m * np.dot(X.T, (h - y)) # Calculate the gradient of the cross-entropy loss with respect to the weights
        return gradient
```

Now when we have an ability to calculate gradient at each step, we can loop it into updating weight and get the final function for gradient descent method. To do so, we take our initial weights firstly, but on every iteration we update it to the opposite direction of gradient by substracting it. If you pay close attention, you can see, that the gradient is being multiplied by learning rate. It is a parameter which controls the step of each update. It helps to control the convergence speed and not allow the algorithm to overshoot the minimum and fail to converge, which is especially crucial on the first steps, when we do not have much information about minimum point and start in random point.


```python
def gradient_descent(self, X, y, w_init):
    w = w_init.copy() # Copy of the initial weights
    for _ in range(self.max_iter):
        gradient = self.gradient(w, X, y)
        w -= self.learning_rate * gradient # Update the weights in the opposite direction of the gradient
    return w
```
