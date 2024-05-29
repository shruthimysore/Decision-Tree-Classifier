import pandas as pd
import numpy as np
import math
from collections import Counter

# Construct the TreeNode class
class TreeNode:
    def __init__(self, attribute=None, value=None, decision=None):
        self.attribute = attribute        
        self.value = value
        self.decision = decision
        self.children = {}
        
    def add_child(self, value, child):
        self.children[value] = child
    
    
# Construct the Decision Tree class
class DecisionTree:
    def __init__(self):
        self.rootnode = None
        
    def fit(self, dataset):
        self.attribute_list = dataset.columns.tolist()
        self.attribute_list.remove('target')
        self.rootnode = self.build_tree(dataset, self.attribute_list)
        return self.rootnode  
        
    def build_tree(self, dataset, attribute_list):
        # If all the target values are the same, return the node with the decision
        if dataset['target'].nunique() == 1:
            return TreeNode(decision=dataset['target'].iloc[0])
        # If the attribute list is empty, return the node with the majority class
        elif len(attribute_list) == 0:
            return TreeNode(decision=dataset['target'].value_counts().idxmax())
        best_attribute = self.find_best_attribute(dataset, attribute_list)
        node = TreeNode(attribute=best_attribute)
        attribute_list.remove(best_attribute)
        for value in dataset[best_attribute].unique():
            # Choose all instances from the dataset with the current value
            sub_dataset = dataset[dataset[best_attribute] == value]
            # If the sub_dataset is empty, return the node with the majority class
            if sub_dataset.shape[0] == 0:
                node.add_child(value, TreeNode(decision=dataset['target'].value_counts().idx_max()))
                return node
            sub_tree = self.build_tree(sub_dataset, attribute_list)
            # Add the sub_tree as a child to the node
            node.add_child(value, sub_tree)
        return node
    
    def find_best_attribute(self, dataset, attribute_list):
        best_attribute = None
        best_gain = 0
        for attribute in attribute_list:
            gain = self.calculate_information_gain(dataset[attribute], dataset['target'])
            if gain > best_gain:
                best_gain = gain
                best_attribute = attribute
        return best_attribute
    
    def calculate_information_gain(self, data, target):
        information_gain = 0
        # Calculate entropy of the target
        target_entropy = self.calculate_entropy(target)
        for value in data.unique():
            # Select target column data where current value is present
            sub_data = target[data == value]
            # Calculate entropy of sub_data
            sub_data_entropy = self.calculate_entropy(sub_data)
            information_gain = target_entropy - sub_data_entropy
        return information_gain
        
    def calculate_entropy(self, data):
        entropy = 0
        # Calculate frequency of each value in the data
        value_frequencies = data.value_counts()
        for frequency in value_frequencies:
            probability = frequency / data.shape[0]
            sum = -(probability * math.log2(probability))
            entropy += sum
        return entropy
            
        
    def predict(self, dataset):
        predictions = []
        for index, row in dataset.iterrows():
            predictions.append(self.predict_row(row))
        return predictions
    
    def predict_row(self, row):
        node = self.rootnode
        while node.decision is None:
            attribute = node.attribute
            value = row[attribute]
            node = node.children[value]
        return node.decision
    
    def compute_accuracy(self, y_test, predictions):
        correct = 0
        for i in range(len(predictions)):
            if predictions[i] == y_test[i]:
                correct += 1
        return correct / len(predictions)
            
        
            
        
        
        