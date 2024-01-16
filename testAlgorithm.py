from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from decison_tree import DecisionTree, calculate_entropy, calculate_gini

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Crea un'istanza dell'albero di decisione
tree = DecisionTree(min_samples_split=2, max_depth=None, impurity='entropy')

# Addestra l'albero di decisione sul training set
tree.fit(X_train, y_train)

# Esegui le previsioni sull'insieme di test
y_pred = tree.predict(X_test)



# Calcola l'accuratezza delle previsioni
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)