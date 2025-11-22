import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import pandas as pd

# TODO: Write a simple Winsorizer transformer which takes a lower and upper quantile and cuts the
# data accordingly

#for a specific column or several columns, calculate the upper and lower quantile
#then, if the observation value is > upper quantile or < lower quantile, 
# replace it with the upper quartile or lower quartile value accordingly 
# this is called clipping of the dataset 
class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, columns, lower_quantile, upper_quantile):
        self.columns = columns
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile


    def fit(self, X, y=None):
        self.lower_quantile_ = X[self.columns].quantile(self.lower_quantile)
        self.upper_quantile_ = X[self.columns].quantile(self.upper_quantile)
        return self

    def transform(self, X):
        X = X.copy()
        
        for col in self.columns:
            X[col] = np.where(X[col] < self.lower_quantile_[col], self.lower_quantile_[col], X[col])
            X[col] = np.where(X[col] > self.upper_quantile_[col], self.upper_quantile_[col], X[col])
            
        return X