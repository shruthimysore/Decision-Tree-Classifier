import numpy  as np
import pandas as pd
from DecisionTree import DecisionTree
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("house_votes_84.csv")
X = dataset.drop('target', axis = 1)
Y = dataset['target']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
dt = DecisionTree()
decision_tree = dt.fit(X_train.join(y_train))
predictions = dt.predict(X_test)
accuracy = dt.compute_accuracy(y_test.tolist(), predictions)
print(y_test)
print(predictions)
print(f"Accuracy: {accuracy}")



