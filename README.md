# Decision Tree Classifier

## Introduction
Welcome to the documentation for the DecisionTree classifier. This code implements a decision tree algorithm for classification tasks. The decision tree is built based on the principles of information gain and impurity measures, such as entropy and Gini impurity.

## Features
* `Decision Tree Algorithm`: The classifier is built using a decision tree algorithm that recursively splits the data based on the features and thresholds that maximize information gain or reduce impurity.
* `Support for Entropy and Gini Impurity`: The classifier supports both entropy and Gini impurity as impurity measures for splitting the data.
* `Customizable Parameters`: You can customize various parameters of the decision tree, such as the minimum number of samples required to split a node, the maximum depth of the tree, and the impurity measure to be used.
* `Efficient Implementation`: The code has been optimized for efficiency, ensuring fast training and prediction times even for large datasets.

## Code Structure
#### The code is organized as follows:

* `calculate_entropy(labels)`: Function to calculate the entropy of a set of labels.
* `calculate_gini(labels)`: Function to calculate the Gini impurity of a set of labels.
* `DecisionNode`: Class representing a node in the decision tree. It contains information about the feature index, threshold, value (for leaf nodes), and the branches for true and false conditions.
* `DecisionTree`: Class implementing the decision tree classifier. It contains methods for splitting the data, calculating impurity, finding the best split, building the tree, fitting the model, and making predictions.

## Usage
To use the DecisionTree classifier, follow these steps:

1) Import the necessary libraries, including numpy.
2) Create an instance of the DecisionTree class, optionally specifying the desired parameters.
3) Fit the model by calling the fit(X, y) method, where X is the input features and y is the corresponding labels.
4) Use the trained model to make predictions by calling the predict(X) method, where X is the input features for prediction.

Here is an example of how to use the DecisionTree classifier:

```import numpy as np
from decision_tree import DecisionTree

# Create the decision tree classifier
classifier = DecisionTree(min_samples_split=5, max_depth=3, impurity='entropy')

# Fit the model to the training data
X_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y_train = np.array([0, 1, 0])
classifier.fit(X_train, y_train)

# Make predictions on new data
X_test = np.array([[2, 3, 4], [5, 6, 7]])
predictions = classifier.predict(X_test)

print("Predictions:", predictions)
```

## Conclusion
The DecisionTree classifier provides a powerful and flexible tool for classification tasks. It leverages the principles of decision tree algorithms and impurity measures to construct a model that can make accurate predictions on new data.
