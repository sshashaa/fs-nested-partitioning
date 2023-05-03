# Feature Selection with Nested Patitioning
## Introduction
Feature selection is a crucial step in machine learning that involves identifying the most relevant features in a dataset. However, when dealing with large datasets, traditional feature selection methods can become computationally expensive and inefficient. To address this issue, simulation optimization using nested partitioning has emerged as a promising technique for feature selection in large datasets. This method involves simulating different feature subsets and optimizing them using nested partitioning algorithms. By doing so, it allows for a more efficient and effective feature selection process, enabling better predictive performance and improved model interpretability. In this article, we will explore the concept of solving feature selection for large datasets using simulation optimization using nested partitioning.

The feature selection problem can be seen as a combinatorial optimization problem, according to the Introduction. To address this problem, we adopt Olafsson and Yang's optimization-based feature selection method [1], which utilizes the nested partitions (NP) metaheuristic for combinatorial optimization. The success of the NP metaheuristic relies on a partitioning scheme that applies an application-specific structure to the search space. Olafsson and Yang have devised a method for intelligently partitioning the feature subset space, and their approach has proven to be effective when compared to other feature selection methods.

When dealing with complex large-scale problems the NP method provides an attractive alternative. Another important class of problems are combinatorial optimization problems (COP) where the feasible region is finite but its size typically grows exponentially in the input parameters of the problem. The NP method is best viewed as a metaheuristic framework, and it has similarities to branching methods in that it creates partitions of the feasible region like branch-and-bound does. However, it also has some unique features that make it well suited for very hard large-scale optimization problems.


To understand the methodology, we need to familiarize ourselves with the nested partition method, which was developed to tackle combinatorial optimization problems. For the feature selection problem, the feasible region is the collection of all possible feature subsets [2].

$\Omega = {{a_1, a_2, ..., a_n \mid a_i \in {0, 1}}}$

where $a_i$ is 1, if the $i^{th}$ attribute is selected.

Since the dataset does not have any class, then we cannot use the information gain formula as it requires a class variable to compute. However, if we have other categorical or continuous variables in the dataset, you could use them to create a split and calculate the information gain for each split. Alternatively, we use another splitting criterion, such as mean squared error (MSE), to split the data and create decision trees. These methods do not require a class variable and can be used for unsupervised learning problems. The formula for MSE is:

$MSE = \frac{1}{n} \sum(y_i - \bar{y})^2$

where n is the number of observations, $y_i$ is the predicted value for the i-th observation, and $\bar{y}$ is the mean of the predicted values for all observations.

## Code Description
### FSwNP.ipynb
This is our first attempt to provide new idea for feature selection by updating the probablities of chooosing a feature in our sampling, later we decided to move to more systematic system by [1], and we did not used this script anymore. The given code contains four functions as described below:

1. mean_squared_error(y, x1, x2, x3, x4): This function takes five input arguments, y is the actual target variable, and x1, x2, x3, and x4 are predictor variables. It calculates the predicted value using the product of x1 and x2 and then computes the mean squared error loss between the predicted and actual values. It returns the computed loss.

2. loss_calculator(df,a): This function takes two input arguments, df is a pandas DataFrame containing the predictor variables x1, x2, x3, x4, and target variable y, and a is a list of four coefficients corresponding to each predictor variable. It updates the DataFrame df with an additional column called Loss, where each row's loss is calculated by applying the mean_squared_error() function using the corresponding coefficients in the a list. The function returns the updated DataFrame.

3. performance_calculator(sampled_df,a,i): This function takes three input arguments, sampled_df is a pandas DataFrame containing the predictor variables x1, x2, x3, x4, and target variable y, a is a list of four coefficients corresponding to each predictor variable, and i is an integer representing the index of the coefficient that needs to be updated. It first updates the ith coefficient to 0, then calls the loss_calculator() function to update the Loss column in the sampled_df DataFrame using the updated a list. It then calculates the mean of the Loss column in the sampled_df DataFrame and returns the mean loss.

4. update_df_itr(df_itr,new_p,k): This function takes three input arguments, df_itr is a pandas DataFrame containing the results of each iteration, new_p is a list containing four updated coefficients, and k is an integer representing the iteration number. It first creates a new list new_itr containing the iteration number k followed by the updated coefficients new_p. It then creates a new pandas Series object using the new_itr list, with column names taken from the df_itr DataFrame. Finally, the function appends this new row to the df_itr DataFrame and returns the updated DataFrame.


As you can see this approach is not object oriented, and is not a good practice. So we later code the problem in a systematic prespective in "intelligence feature selection by Olafson.ipynb".

### intelligence feature selection by Olafson.ipynb
This script mimics the algorithm provided by [1] in python. For furthur information about the functions and objective we reffere you to read the paper by Ólafsson and Yang, sicne we used the exact terminalogy used in their paper.

### Greedy Feature Selection.ipynb
First of all, we should confess that the name could have been selected more carefully, since our proposed algorithm is not greedy heuristic, and it uses the modified version of [1] using rapid screening. This script provides the code for feature selection using Intelliegence Nested Patitioning with Rapid Screening. To do so we first run the "RapindScreeing.py" in order to be get the intelligent sence of the features whcih are our ture features and should be included in prediction model. We will go through the functions of each file one-by-one in one example to make it clear for the reader.

First step is checking the entropy;
![alt text](https://github.com/sshashaa/fs-nested-partitioning/blob/main/Figures/entropy.PNG)


## Refrences
[1] S. Ólafsson and J. Yang, “Intelligent partitioning for feature selection,” INFORMS Journal on Computing, vol. 17, no. 3, pp. 339–355, 2005.
[2] J. Pichitlamken and B. L. Nelson, “A combined procedure for optimization via simulation,” ACM Transactions on Modeling and Computer Simulation (TOMACS), vol. 13, no. 2, pp. 155–179, 2003
