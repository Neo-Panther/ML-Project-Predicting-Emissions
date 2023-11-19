import numpy as np
import pandas as pd

class Node:
  def __init__(self, x, y, grad, hess, depth = 6, gamma = 0, min_child_weight = 1, lambda_ = 1, colsample = 1):
      self.x = x
      self.y = y
      self.grad = grad
      self.hess = hess
      self.depth = depth
      self.gamma = gamma
      self.lambda_ = lambda_
      self.min_child_weight = min_child_weight
      self.colsample = colsample
      self.cols = np.random.permutation(x.shape[1])[:round(colsample * x.shape[1])]
      self.sim_score = self.similarity_score([True]*x.shape[0])
      self.gain = float("-inf")
      
      self.split_col = None
      self.split_row = None
      self.lhs_tree = None
      self.rhs_tree = None
      self.pivot = None
      self.val = None
      # making split
      self.split_node()
      
      if self.is_leaf:
          self.val = - np.sum(grad) / (np.sum(hess) + lambda_)
      
  
  def split_node(self):
    self.find_split()
    
    # checking whether it's a leaf or not
    if self.is_leaf:
      return
    
    x = self.x[:, self.split_col]
    lhs = x <= x[self.split_row]
    rhs = x > x[self.split_row]
    
    # creating further nodes recursivly
    self.lhs_tree = Node(
      self.x[lhs],
      self.y[lhs],
      self.grad[lhs],
      self.hess[lhs],
      depth = self.depth - 1,
      gamma = self.gamma,
      min_child_weight = self.min_child_weight,
      lambda_ = self.lambda_,
      colsample = self.colsample
    )
    
    self.rhs_tree = Node(
      self.x[rhs],
      self.y[rhs],
      self.grad[rhs],
      self.hess[rhs],
      depth = self.depth - 1,
      gamma = self.gamma,
      min_child_weight = self.min_child_weight,
      lambda_ = self.lambda_,
      colsample = self.colsample
    )
    
  def find_split(self):
    # iterate through every feature and row
    for c in self.cols:
      x = self.x[:, c]
      for row in range(self.x.shape[0]):
        pivot= x[row]
        lhs = x <= pivot
        rhs = x > pivot
        sim_lhs = self.similarity_score(lhs)
        sim_rhs = self.similarity_score(rhs)
        gain = sim_lhs + sim_rhs - self.sim_score - self.gamma
        
        if gain < 0 or self.not_valid_split(lhs) or self.not_valid_split(rhs):
          continue
        
        if gain > self.gain:
          self.split_col = c
          self.split_row = row
          self.pivot = pivot
          self.gain = gain
                  
  def not_valid_split(self, masks):
    if np.sum(self.hess[masks]) < self.min_child_weight:
      return True
    return False
  
  @property
  def is_leaf(self):
    if self.depth < 0 or self.gain == float("-inf"):
      return True
    return False
              
  def similarity_score(self, masks):
    return np.sum(self.grad[masks]) ** 2 / ( np.sum(self.hess[masks]) + self.lambda_ )
  
  
  def predict(self, x):
    return np.array([self.predict_single_val(row) for row in x])
  
  def predict_single_val(self, x):
    if self.is_leaf:
      return self.val
    
    return self.lhs_tree.predict_single_val(x) if x[self.split_col] <= self.pivot else self.rhs_tree.predict_single_val(x)

class XGBTree:
  def __init__(self, x, y, grad, hess, depth = 6, gamma = 0, min_child_weight = 1, lambda_ = 1, colsample = 1, subsample = 1):
    indices = np.random.permutation(x.shape[0])[:round(subsample * x.shape[0])]
    
    self.tree = Node(
      x[indices],
      y[indices],
      grad[indices],
      hess[indices],
      depth = depth,
      gamma = gamma,
      min_child_weight = min_child_weight,
      lambda_ =  lambda_,
      colsample = colsample,
    )
  
  def predict(self, x):
    return self.tree.predict(x)
    
class XGBRegressor:
  def __init__(self, eta = 0.3, n_estimators = 100, max_depth = 6, gamma = 0, min_child_weight = 1, lambda_ = 1, colsample = 1, subsample = 1):
    self.eta = eta
    self.n_estimators = n_estimators
    self.max_depth = max_depth
    self.gamma = gamma
    self.min_child_weight = min_child_weight
    self.lambda_ = lambda_
    self.colsample = colsample
    self.subsample = subsample
    self.history = {
        "train" : list(),
        "test" : list()
    }
    
    # list of all weak learners
    self.trees = list()
    
    self.base_pred = None
      
  def fit(self, x, y, eval_set = None):
    # checking Datatypes
    if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
      x = x.values
    if not isinstance(x, np.ndarray):
        raise TypeError("Input should be pandas Dataframe/Series or numpy array.")
        
    if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
        y = y.values
    if not isinstance(y, np.ndarray):
        raise TypeError("Input should be pandas Dataframe/Series or numpy array.")
    
    
    
    base_pred = np.full(y.shape, np.mean(y)).astype("float64")
    self.base_pred = np.mean(y)
    for n in range(self.n_estimators):
        grad = self.grad(y, base_pred)
        hess = self.hess(y, base_pred)
        estimator = XGBTree(
            x,
            y,
            grad,
            hess,
            depth = self.max_depth,
            gamma = self.gamma,
            min_child_weight = self.min_child_weight,
            lambda_ = self.lambda_,
            colsample = self.colsample,
            subsample = self.subsample
        )
        base_pred = base_pred + self.eta * estimator.predict(x)
        self.trees.append(estimator)
        
        if eval_set:
            X = eval_set[0]
            Y = eval_set[1]
            cost = np.sqrt(np.mean(self.loss(Y, self.predict(X))))
            self.history["test"].append(cost)
            print(f"[{n}] validation_set-rmse : {cost}", end="\t")
        
        cost = np.sqrt(np.mean(self.loss(y, base_pred)))
        self.history["train"].append(cost)
        print(f"[{n}] train_set-rmse : {cost}")
          
  def predict(self, x):
      base_pred = np.full((x.shape[0],), self.base_pred).astype("float64")
      for tree in self.trees:
          base_pred += self.eta * tree.predict(x)
      
      return base_pred
  
  def loss(self, y, a):
      return (y - a)**2
  
  def grad(self, y, a):
      # for 0.5 * (y - a)**2
      return a - y
  
  def hess(self, y, a):
      # for 0.5 * (y - a)**2
      return np.full((y.shape), 1)