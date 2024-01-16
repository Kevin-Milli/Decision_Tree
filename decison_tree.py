import numpy as np

def calculate_entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def calculate_gini(labels):
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    gini_impurity = 1 - np.sum(probabilities ** 2)
    return gini_impurity

class DecisionNode:
    def __init__(self, feature_index=None, threshold=None, value=None, true_branch=None, false_branch=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=None, impurity='entropy'):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.impurity = impurity
        self.root = None

    def split_data(self, X, y, feature_index, threshold):
        true_indices = X[:, feature_index] <= threshold
        false_indices = ~true_indices
        X_true, y_true = X[true_indices], y[true_indices]
        X_false, y_false = X[false_indices], y[false_indices]
        return X_true, y_true, X_false, y_false

    def calculate_impurity(self, y):
        if self.impurity == 'entropy':
            return calculate_entropy(y)
        elif self.impurity == 'gini':
            return calculate_gini(y)

    def calculate_best_split(self, X, y):
        best_gain = -float('inf')
        best_feature_index, best_threshold = None, None

        for feature_index in range(X.shape[1]):
            unique_values = np.unique(X[:, feature_index])
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2

            for threshold in thresholds:
                X_true, y_true, X_false, y_false = self.split_data(X, y, feature_index, threshold)
                impurity_true = self.calculate_impurity(y_true)
                impurity_false = self.calculate_impurity(y_false)
                p_true = len(y_true) / len(y)
                p_false = len(y_false) / len(y)
                gain = self.calculate_impurity(y) - (p_true * impurity_true + p_false * impurity_false)

                if gain > best_gain:
                    best_gain = gain
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold

    def build_tree(self, X, y, depth):
        if len(np.unique(y)) == 1:
            return DecisionNode(value=np.unique(y)[0])

        if len(y) < self.min_samples_split or depth == self.max_depth:
            values, counts = np.unique(y, return_counts=True)
            return DecisionNode(value=values[np.argmax(counts)])

        best_feature_index, best_threshold = self.calculate_best_split(X, y)
        X_true, y_true, X_false, y_false = self.split_data(X, y, best_feature_index, best_threshold)
        true_branch = self.build_tree(X_true, y_true, depth + 1)
        false_branch = self.build_tree(X_false, y_false, depth + 1)

        return DecisionNode(feature_index=best_feature_index, threshold=best_threshold,
                            true_branch=true_branch, false_branch=false_branch)

    def fit(self, X, y):
        self.root = self.build_tree(X, y, depth=0)

    def predict_sample(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self.predict_sample(x, node.true_branch)
        else:
            return self.predict_sample(x, node.false_branch)

    def predict(self, X):
        predictions = []
        for x in X:
            predictions.append(self.predict_sample(x, self.root))
        return np.array(predictions)
    