#Author: Junfu Cheng
#Organization: University of Florida
import numpy as np

class StandScalar:
    def __init__(self, feature_num):
        self.feature_num = feature_num
        self.means = np.zeros((1, feature_num))
        self.stds = np.ones((1, feature_num))
        


    def fit_transfrom(self, X):
        """
        Fit the StandScalar to the data X and then transform it.
        
        Parameters:
        - X: (n_samples, n_features) numpy array of data to fit and transform.
        
        Returns:
        - X_scaled: (n_samples, n_features) numpy array of standardized data.
        """
        # Calculate means and standard deviations for each feature
        self.means = np.mean(X, axis=0, keepdims=True)
        self.stds = np.std(X, axis=0, keepdims=True, ddof=0)  # ddof=0 for population std deviation
        

    def transform(self, X):
        # Avoid division by zero if std is 0
        self.stds[self.stds == 0] = 1
        
        # Standardize the data
        X_scaled = (X - self.means) / self.stds
        
        return X_scaled
    
    def inverse_transform(self, X_scaled):
        """
        Inverse transform standardized data to the original scale using the fitted parameters.
        
        Parameters:
        - X_scaled: (n_samples, n_features) numpy array of standardized data to inverse transform.
        
        Returns:
        - X: (n_samples, n_features) numpy array of original scale data.
        """
        # Avoid division by zero if std is 0
        self.stds[self.stds == 0] = 1
        
        # Inverse transform the data
        X = (X_scaled * self.stds) + self.means
        
        return X
