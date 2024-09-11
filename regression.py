#Author: Junfu Cheng
#Organization: University of Florida
import numpy as np
class Regression:
    def __init__(self, feature_num, bias, X_use_target):
        self.feature_num = feature_num
        self.bias = bias
        self.X_use_target = X_use_target
        self.weights = self.initialize_weights(feature_num, init_type='random')
        
    def predict(self, transform_X, transform_t, X_row):
        feature_num = X_row.shape[1]
        X_scaled = transform_X.transform(X_row)
        prediction_scaled = 0
        for k in range(feature_num):
            prediction_scaled += self.weights[k][0] * X_scaled[0][k]
        prediction = transform_t.inverse_transform(prediction_scaled)
        return prediction
        
    def MSE(self, transform_X, transform_t, X, t):
        sample_num = X.shape[0]
        feature_num = X.shape[1]
        if feature_num != self.feature_num:
            print(feature_num)
            raise ValueError(f'Matrix X must have {self.feature_num} feature(s).')
        cost = 0
        for i in range(sample_num):
            prediction = self.predict(transform_X, transform_t, X[i][:].reshape(1, -1))
            cost += (t[i][0] - prediction)**2
        cost = cost / (2 * sample_num)
        return cost
    
    def cross_validate(self,  X_train, t_train, fold_num):
        """
        Perform k-fold cross-validation and return training and validation folds.
        
        Parameters:
        - X_train: (n, m) numpy array of feature values.
        - t_train: (n, 1) numpy array of target values.
        - fold_num: Number of folds for cross-validation.
        
        Returns:
        - X_train_folds: List of training feature folds.
        - t_train_folds: List of training target folds.
        - X_val_folds: List of validation feature folds.
        - t_val_folds: List of validation target folds.
        """
        # Number of samples
        n_samples = X_train.shape[0]
        
        # Shuffle indices
        indices = np.random.permutation(n_samples)
        
        # Split indices into folds
        fold_size = n_samples // fold_num
        folds = [indices[i * fold_size:(i + 1) * fold_size] for i in range(fold_num)]
        
        # Create lists to hold folds for training and validation
        X_train_folds = []
        t_train_folds = []
        X_val_folds = []
        t_val_folds = []
        
        # Generate folds
        for i in range(fold_num):
            # Validation indices
            val_indices = folds[i]
            
            # Training indices (all other indices)
            train_indices = np.concatenate([folds[j] for j in range(fold_num) if j != i])
            
            # Split data into training and validation sets
            X_train_fold = X_train[train_indices]
            t_train_fold = t_train[train_indices].reshape(-1, 1)
            X_val_fold = X_train[val_indices]
            t_val_fold = t_train[val_indices].reshape(-1, 1)
            
            # Append to lists
            X_train_folds.append(X_train_fold)
            t_train_folds.append(t_train_fold)
            X_val_folds.append(X_val_fold)
            t_val_folds.append(t_val_fold)
        
        return X_train_folds, t_train_folds, X_val_folds, t_val_folds

    def initialize_weights(self, num_weights, init_type='random', seed=5):
        """
        Initialize weights for the LMS algorithm.
        
        Parameters:
        - num_weights: Number of weights to initialize.
        - init_type: Type of initialization ('zeros', 'ones', 'random').
        - seed: Optional random seed for reproducibility.
        
        Returns:
        - weights: Initialized weights as a NumPy array.
        """
        if seed is not None:
            np.random.seed(seed)
        
        if init_type == 'zeros':
            weights = np.zeros(num_weights)
        elif init_type == 'ones':
            weights = np.ones(num_weights)
        elif init_type == 'random':
            weights = np.random.randn(num_weights)  # Normally distributed random weights
        else:
            raise ValueError("Invalid init_type. Choose from 'zeros', 'ones', 'random'.")
            
        weights = weights.reshape(-1, 1)
        
        return weights
