from functools import wraps

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.metrics import make_scorer
from sklearn import cross_validation as cv
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.manifold import TSNE

def score(*args, **kwargs):
  """Decorator, that transform a function to a scorer.
  A scorer has the arguments estimator, X, y_true, sample_weight=None
  """
  decorator_args = args
  decorator_kwargs = kwargs
  def score_decorator(func):
    @wraps(func)
    def func_wrapper(*args, **kwargs):
      func_args = args
      func_kwargs = kwargs
      scorer = make_scorer(func, *decorator_args, **decorator_kwargs)
      return scorer(*func_args, **func_kwargs)
    return func_wrapper
  return score_decorator

def folds(y, n_folds=4, **kwargs):
  return cv.KFold(n=len(y), n_folds=n_folds, shuffle=True, random_state=42, **kwargs)

def cross_val(estimator, X, y, n_jobs=-1, **kwargs):
  # Extract values from pandas DF
  if 'values' in X:
    X = X.values
    if 'values' in y:
      y = y.values
  # Return Cross validation score
  return cv.cross_val_score(estimator, X, y, cv=folds(y), n_jobs=n_jobs, **kwargs)


class BaseTransform(BaseEstimator, ClassifierMixin, TransformerMixin):
  """Transform Interface"""
  def __init__(self):
    pass

  def fit(self, X, y=None, **fit_params):
    return self

  def transform(self, X):
    return X


class PandasTransform(BaseTransform):
  def __init__(self):
    pass

  def transform(self, X):
    return X.values


class Log1pTransform(BaseTransform):
  def __init__(self, columns=None):
    self.columns = columns=None

  def transform(self, X):
    if self.columns:
      for column in self.columns:
        X[column] = np.log1p(X[column])
        return X
    else:
      return np.log1p(X)

  def inverse_transform(self, X):
    if self.columns:
      for column in self.columns:
        X[column] = np.expm1(X[column])
        return X
    else:
      return np.expm1(X)


class NanPreProcessor(BaseTransform):
  def __init__(self, columns=None, nan=None):
    self.columns = columns
    self.nan = nan

  def transform(self, X):
    X = X.copy()
    if self.columns:
      for col, nan in self.columns:
        if col in X.columns:
          X[col].fillna(nan, inplace=True)
    if self.nan is not None:
      X.fillna(self.nan, inplace=True)
    return X

def tsne_plot(X, y, title="", metric='l1', random_state=0, legend_loc='upper left', n_samples=None, n_components=2):
  """Plots the first 2 components of the t-distributed Stochastic Neighbor Embedding
  References:
   * http://blog.kaggle.com/2012/11/02/t-distributed-stochastic-neighbor-embedding-wins-merck-viz-challenge/"""

  if n_samples:
      # Get the shape of the training set
      n_samples_orig, n_features = np.shape(X)

      # Select 5000 random indices
      rnd_indices = np.random.choice(n_samples_orig, n_samples)

      X = X[rnd_indices]
      y = y[rnd_indices]

  # Create a t-SNE model
  model = TSNE(n_components=n_components, random_state=random_state, metric=metric)
  X_trans = model.fit_transform(X)

  # Get a list of unique labels
  labels = np.unique(y)

  # This is only needed to adjust the size of the figure
  # because otherwise it is really small
  plt.figure(figsize=(15, 15), dpi=120)

  # Get a list of color values
  colors = cm.rainbow(np.linspace(0, 1, len(labels)))

  # Loop over labels
  # enumerate also return the index from the list
  for i, label in enumerate(labels):

      # Get a feature vector with the matching label
      # and add a scatter plot with the dataset
      plt.scatter(X_trans[y == label][:,0], X_trans[y == label][:,1], c=colors[i], label=label)

  # Add a legend
  plt.legend(loc=legend_loc)

  # Add axis labels
  plt.xlabel("1st component")
  plt.ylabel("2nd component")

  # Add a title
  plt.title(title)

  # Render the plot
  plt.show()