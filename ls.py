#Author: Junfu Cheng
#Organization: University of Florida
from regression import Regression
import numpy as np
class LS(Regression):
    def __init__(self, feature_num):
        super().__init__(feature_num)
        
    def cross_validate_fit(self, transform_X, transform_t,  X_train, t_train, X_test, t_test, fold_num = 5, hyperparameters = [0.001, 0.01, 0.1, 1.0, 10.0, 100]):
        
        X_train_folds, t_train_folds, X_val_folds, t_val_folds = self.cross_validate(  X_train,t_train, fold_num)
        for lamb in hyperparameters:
            aveCost = 0
            for X_train_fold, t_train_fold, X_val_fold, t_val_fold in zip(X_train_folds, t_train_folds, X_val_folds, t_val_folds):
                self.fit(transform_X, transform_t, X_train_fold, t_train_fold, lamb = lamb)
                cost = self.MSE(transform_X, transform_t, X_val_fold, t_val_fold)
                aveCost += cost
            aveCost = aveCost / len(hyperparameters)
            testCost = self.MSE(transform_X, transform_t, X_test, t_test)
            print("Report from LS::cross_validate_fit:")
            print(f'Value of lamb: {lamb}, Value of cost in valid dataset: {aveCost},Value of cost in test dataset: {aveCost} ')
            
        

        
    def fit(self, transform_X, transform_t, X, t, lamb = 0.1):
        X = transform_X.transform(X)
        t = transform_t.transform(t)
        
        R = self.calculate_R(X)
        P = self.calculate_P(X, t)
        
        # Create the identity matrix of the same size as R
        E = np.eye(R.shape[0])
        
        # Compute lambda * E
        lambda_E = lamb * E
        
        # Compute the inverse of (R - lambda * E)
        matrix_to_multiply = np.linalg.inv(R + lambda_E)
        
        # Compute the product of the inverse of R and P
        self.weights = matrix_to_multiply @ P
        
        
    def calculate_R(self, X):
        sample_num = X.shape[0]
        feature_num = X.shape[1]
        R = np.zeros((feature_num, feature_num))
        for k in range(feature_num):
            for j in range(feature_num):
                for i in range(sample_num):
                    R[k][j] += X[i][k]*X[i][j]
                R[k][j] = R[k][j] / sample_num
                    
        return R
    
    def calculate_P(self, X, t):
        sample_num = X.shape[0]
        feature_num = X.shape[1]
        P = np.zeros((feature_num, 1))
        for j in range(feature_num):
            for i in range(sample_num):
                P[j][0] += X[i][j] * t[i][0]
            P[j][0] = P[j][0] / sample_num
        
        return P
