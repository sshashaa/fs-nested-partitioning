# Feature Selection with Nested Patitioning
## Introduction
Feature selection is a crucial step in machine learning that involves identifying the most relevant features in a dataset. However, when dealing with large datasets, traditional feature selection methods can become computationally expensive and inefficient. To address this issue, simulation optimization using nested partitioning has emerged as a promising technique for feature selection in large datasets. This method involves simulating different feature subsets and optimizing them using nested partitioning algorithms. By doing so, it allows for a more efficient and effective feature selection process, enabling better predictive performance and improved model interpretability. In this article, we will explore the concept of solving feature selection for large datasets using simulation optimization using nested partitioning.

The feature selection problem can be seen as a combinatorial optimization problem, according to the Introduction. To address this problem, we adopt Olafsson and Yang's optimization-based feature selection method \cite{olafsson2005intelligent}, which utilizes the nested partitions (NP) metaheuristic for combinatorial optimization. The success of the NP metaheuristic relies on a partitioning scheme that applies an application-specific structure to the search space. Olafsson and Yang \cite{olafsson2005intelligent} have devised a method for intelligently partitioning the feature subset space, and their approach has proven to be effective when compared to other feature selection methods.

To understand the methodology, we need to familiarize ourselves with the nested partition method, which was developed to tackle combinatorial optimization problems. For the feature selection problem, the feasible region is the collection of all possible feature subsets \cite{pichitlamken2003combined}.

$\Omega = \{{a_1, a_2, ..., a_n \mid a_i \in {0, 1}}\}$

where $a_i$ is 1, if the $i^{th}$ attribute is selected.

Olafson and Yang use their method on a categorical dataset in which each data point is categorized in a class $c$. 

Since the dataset does not have any class, then we cannot use the information gain formula as it requires a class variable to compute. However, if we have other categorical or continuous variables in the dataset, you could use them to create a split and calculate the information gain for each split. Alternatively, we use another splitting criterion, such as Gini impurity or mean squared error, to split the data and create decision trees. These methods do not require a class variable and can be used for unsupervised learning problems.
The formula for Gini impurity is:

$Gini(p) = \sum_{i=1}^{J} p_i(1-p_i)$

where J is the number of classes, and p is the probability of observing each class in a particular subset of the data.

The formula for mean squared error is:

$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \bar{y})^2$

where n is the number of observations, $y_i$ is the predicted value for the i-th observation, and $\bar{y}$ is the mean of the predicted values for all observations.

This repository will provide the code for feature selection using Intelliegence Nested Patitioning with Rapid Screening.
