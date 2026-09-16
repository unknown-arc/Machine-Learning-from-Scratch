# Machine Learning From Scratch

A collection of **Machine Learning algorithms implemented from scratch using mathematics and Python**, with their results compared against equivalent models from **scikit-learn**.

The main goal is to understand how ML algorithms work internally instead of using only ready-made libraries.

## Overview

Each model focuses on:

* Mathematical foundation
* From-scratch Python implementation
* Training and prediction
* Model evaluation
* Comparison with `scikit-learn`

## Models

### 1. Linear Regression

Used for predicting continuous values.

**Linear Model:**

$$
\hat{y} = X\theta
$$

**Mean Squared Error:**

$$
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

**Normal Equation:**

$$
\theta=(X^TX)^{-1}X^Ty
$$

**scikit-learn:** `LinearRegression`

---

### 2. Logistic Regression

Used for binary classification.

**Sigmoid Function:**

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

where:

$$
z=X\theta+b
$$

**Binary Cross-Entropy Loss:**

$$
J(\theta)=-\frac{1}{n}\sum_{i=1}^{n}
[y_i\log(\hat{y}_i)+(1-y_i)\log(1-\hat{y}_i)]
$$

**scikit-learn:** `LogisticRegression`

---

### 3. K-Nearest Neighbors (KNN)

Classifies a sample based on its nearest neighbors.

**Euclidean Distance:**

$$
d(x,y)=\sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}
$$

The predicted class is generally the majority class among the \(k\) nearest samples.

**scikit-learn:** `KNeighborsClassifier`

---

### 4. Decision Tree

Builds a tree by recursively splitting data based on feature values.

**Entropy:**

$$
H(S)=-\sum_{i=1}^{c}p_i\log_2(p_i)
$$

**Information Gain:**

$$
IG(S,A)=H(S)-
\sum_{v\in Values(A)}
\frac{|S_v|}{|S|}H(S_v)
$$

**scikit-learn:** `DecisionTreeClassifier`

---

### 5. Naive Bayes

A probabilistic classification algorithm based on Bayes' theorem.

**Bayes' Theorem:**

$$
P(C|X)=\frac{P(X|C)P(C)}{P(X)}
$$

With the conditional independence assumption:

$$
P(C|X)\propto P(C)\prod_{i=1}^{n}P(x_i|C)
$$

**scikit-learn:** `GaussianNB`

---

### 6. K-Means Clustering

An unsupervised learning algorithm that divides data into \(k\) clusters.

**Euclidean Distance:**

$$
d(x,\mu)=\sqrt{\sum_{i=1}^{n}(x_i-\mu_i)^2}
$$

**Centroid Update:**

$$
\mu_k=\frac{1}{|C_k|}
\sum_{x_i\in C_k}x_i
$$

**Objective Function:**

$$
J=\sum_{k=1}^{K}\sum_{x_i\in C_k}
\|x_i-\mu_k\|^2
$$

**scikit-learn:** `KMeans`

---

### 7. Random Forest

An ensemble learning algorithm that combines multiple Decision Trees.

For classification, the final prediction is generally based on majority voting:

$$
\hat{y} = \text{mode}(T_1(x), T_2(x), \ldots, T_B(x))
$$

where $T_i$ represents an individual Decision Tree.

**scikit-learn:** `RandomForestClassifier`

---

## Comparison

| Model               | From Scratch                  | scikit-learn             |
| ------------------- | ----------------------------- | ------------------------ |
| Linear Regression   | Mathematical implementation   | `LinearRegression`       |
| Logistic Regression | Sigmoid + Gradient Descent    | `LogisticRegression`     |
| KNN                 | Distance-based implementation | `KNeighborsClassifier`   |
| Decision Tree       | Entropy + Information Gain    | `DecisionTreeClassifier` |
| Naive Bayes         | Bayes Theorem                 | `GaussianNB`             |
| K-Means             | Centroid-based clustering     | `KMeans`                 |
| Random Forest       | Multiple Decision Trees       | `RandomForestClassifier` |

## Technologies

* Python
* NumPy
* Pandas
* Matplotlib
* scikit-learn
* Jupyter Notebook

## Purpose

This project is created for **understanding Machine Learning algorithms from their mathematical foundations**.

Instead of directly using ML libraries, the algorithms are first implemented from scratch and then compared with optimized `scikit-learn` implementations.

## Future Work

Possible algorithms to add:

* Support Vector Machine
* PCA
* Gradient Boosting
* Neural Networks
* DBSCAN
* AdaBoost

## Author

**ARC**

Machine Learning / Data Analytics Learning Project

