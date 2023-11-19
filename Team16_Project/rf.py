from dt import DecisionTree
import numpy as np

class RandomForest():
    def __init__(self, n_estimators, sample_size, min_leaf=3, max_depth=10):
        self.n_estimators = n_estimators
        self.sample_size = sample_size
        self.min_leaf = min_leaf
        self.max_depth = max_depth
        self.trees = None

    def create_tree(self, x, y):
        idxs = np.random.randint(len(y), size=self.sample_size)
        dt = DecisionTree(self.max_depth, self.min_leaf)
        dt.fit(x[idxs], y[idxs])
        return dt

    def fit(self, x, y):
        self.trees = None
        self.trees = [self.create_tree(x, y) for i in range(self.n_estimators)]

    def predict(self, x):
        tree_predictions = [tree.predict(x) for tree in self.trees]
        prediction = np.mean(tree_predictions, axis=0)
        return prediction