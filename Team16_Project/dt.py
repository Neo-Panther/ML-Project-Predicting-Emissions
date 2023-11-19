import numpy as np
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf_node(self):
        return self.value is not None
class DecisionTree:
    def __init__(self, max_depth=20, min_samples=10):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.tree = []
    
    def fit(self, X, y):
        self.tree = self.grow_tree(X, y)
    
    def predict(self, X):
        return np.array([self.travers_tree(x, self.tree) for x in X])
    
    def most_common(self, y):
        return np.sum(y) / len(y)
    
    def entropy(self, y):  
        predict = np.sum(y) / len(y)
        mse = np.sum((predict - y)**2) / len(y)
        mae = np.sum(np.abs(predict - y)) / len(y)
        return mae
    
    def best_split(self, X, y):
        best_feature, best_threshold = None, None
        best_gain = -1
        
        for i in range(X.shape[1]):
            thresholds = np.unique(X[:, i])
            for threshold in thresholds:
                gain = self.information_gain(X[:, i], y, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = i
                    best_threshold = threshold
        return best_feature, best_threshold
    
    def information_gain(self, X_column, y, threshold):       
        n = len(y)
        parent = self.entropy(y)
        
        left_indexes = np.argwhere(X_column <= threshold).flatten()
        right_indexes = np.argwhere(X_column > threshold).flatten()
        
        child = 0 
        
        if len(left_indexes) != 0:
            e_l, n_l = self.entropy(y[left_indexes]), len(left_indexes)
            child += (n_l / n) * e_l
        if len(right_indexes) != 0:
            e_r, n_r = self.entropy(y[right_indexes]), len(right_indexes)
            child += (n_r / n) * e_r
            
        return parent - child
    
    def grow_tree(self, X, y, depth=0):
        n_samples = X.shape[0]
        
        if n_samples <= self.min_samples or depth >= self.max_depth:
            return Node(value=self.most_common(y))
        
        best_feature, best_threshold = self.best_split(X, y)
        
        left_indexes = np.argwhere(X[:, best_feature] <= best_threshold).flatten()
        right_indexes = np.argwhere(X[:, best_feature] > best_threshold).flatten()
        
        if len(left_indexes) == 0 or len(right_indexes) == 0:
            return Node(value=self.most_common(y))
        
        left = self.grow_tree(X[left_indexes, :], y[left_indexes], depth+1)
        right = self.grow_tree(X[right_indexes, :], y[right_indexes], depth+1)
        
        return Node(best_feature, best_threshold, left, right)
    
    def travers_tree(self, x, tree):
        if tree.is_leaf_node():
            return tree.value
        
        if x[tree.feature] <= tree.threshold:
            return self.travers_tree(x, tree.left)
        return self.travers_tree(x, tree.right)