#Author: Junfu Cheng
#Organization: University of Florida
from regression import Regression
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
class LMS(Regression):
    def __init__(self, feature_num, bias, X_use_target):
        super().__init__(feature_num, bias, X_use_target)
        
        
        
    def fit(self, transform_X, transform_t, X, t,X_val, t_val, X_test, t_test, epoch = 300, lr =  0.0001):
        weights_list = []
        cost_list = []
        for i in tqdm(range(epoch), desc="Processing"):
            self.weights = self.step( transform_X, transform_t, X, t, lr = lr)
            weights_list.append(self.weights)
            cost = self.MSE(transform_X, transform_t, X, t)
            cost_list.append(cost[0][0])
            #print(f'Epoch: {i}, train cost: {cost}, val cost: {self.MSE(transform_X, transform_t, X_val, t_val)}, test cost: {self.MSE(transform_X, transform_t, X_test, t_test)}')
        # Create a plot
        x_values = list(range(len(cost_list)))
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, cost_list, marker='o', linestyle='-', color='b')

        # Add titles and labels
        plt.title('Cost Curve')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')

        # Show grid
        plt.grid(True)

        # Display the plot
        plt.show()
        return weights_list

        
    def step(self, transform_X, transform_t, X, t, lr = 0.001):
        e = self.calculate_error(transform_X, transform_t, X, t)
        X_scaled = transform_X.transform(X)
        new_weights = self.weights.T + lr*e.T@X_scaled
        new_weights = new_weights.T
        return new_weights

    def calculate_error(self, transform_X, transform_t, X, t):
        sample_num = X.shape[0]
        feature_num = X.shape[1]
        num_weights = feature_num
        if feature_num != self.feature_num:
            print(feature_num)
            raise ValueError(f'Matrix X must have {self.feature_num} feature(s).')
        error = 0
        X_scaled = transform_X.transform(X)
        t_scaled = transform_t.transform(t)
        
        
        
        new_weights = np.zeros(num_weights).reshape(-1, 1)
        for i in range(feature_num):
            ds = X_scaled@self.weights
        e = t_scaled - ds
        return e

    def profile(self, weights_list):
        weights_array = np.array([w.flatten() for w in weights_list])  # shape will be (number_of_iterations, 16)

        # Generate x values (e.g., iteration numbers)
        iterations = list(range(len(weights_list)))

        # Plot each weight
        plt.figure(figsize=(14, 8))

        if self.bias == True:
            for i in range(weights_array.shape[1]-1):  # Iterate over each weight
                plt.plot(iterations, weights_array[:, i], label=f'Weight {i+1}: {list(dataset.X_dict.keys())[i]}')

            plt.plot(iterations, weights_array[:, weights_array.shape[1]-1], label=f'Weight {weights_array.shape[1]}: bias')
        else:
            for i in range(weights_array.shape[1]):  # Iterate over each weight
                plt.plot(iterations, weights_array[:, i], label=f'Weight {i+1}: {list(dataset.X_dict.keys())[i]}')


        # Add titles and labels
        plt.title('Change in Weights Over Iterations', fontsize=14)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Weight Value', fontsize=12)

        # Add a legend
        plt.legend(loc='best')

        # Show grid
        plt.grid(True)

        # Display the plot
        plt.show()
