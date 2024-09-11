#Author: Junfu Cheng
#Organization: University of Florida
import numpy as np

class Dataset:
    def __init__(self, file_path, dictionary, target_name, bias, X_use_target):
        self.data = self.parse_file(file_path)
        self.dictionary = dictionary
        self.target_name = target_name
        self.bias = bias
        self.X_use_target = X_use_target

        self.t, self.X , self.X_dict = self.set_target(target_name, bias= bias, X_use_target = X_use_target)
        self.X_train, self.t_train, self.X_test, self.t_test = self.train_test_split(self.t, self.X)

    def train_test_split(self, t, X, training_partial = 2/3):
        # Shuffle rows
        indices = np.random.permutation(X.shape[0])
        X = X[indices]
        # Shuffle rows
        t = t[indices].reshape(-1, 1)
        # Determine the split index
        split_index = int(training_partial * X.shape[0])

        # Split the arrays
        X_train = X[:split_index]  # Training features
        t_train = t[:split_index].reshape(-1, 1)  # Training target
        X_test = X[split_index:]   # Testing features
        t_test = t[split_index:].reshape(-1, 1)   # Testing target
        return X_train, t_train, X_test, t_test

    
        
    
    def set_target(self, target_name, bias = True, X_use_target = False):
        if target_name not in self.dictionary:
            raise KeyError(f"Key '{target_name}' not found in the dictionary.")
        keys_list = list(self.dictionary.keys())
        target_index = keys_list.index(target_name)

        t = self.data[:, target_index].reshape(-1, 1)
        if X_use_target == False:
            X = np.delete(self.data, target_index, axis = 1)
            X_dict = {k: v for k, v in self.dictionary.items() if k != target_name}
        else:
            X = self.data
            X_dict = self.dictionary
        if bias == True:
            # Create a column of ones with the same number of rows as the original array
            ones_column = np.ones((X.shape[0], 1))
            # Append the column of ones to the original array
            X = np.hstack((X, ones_column))

       
        return t, X, X_dict 

    

    # Function to parse the file content
    def parse_file(self, file_path):
        # Read the file content
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Initialize lists to store the data
        data = []
        
        # Process the lines in pairs
        for i in range(0, len(lines), 2):
            # First line with main values
            main_values = lines[i].strip().split()
            # Second line with additional values
            additional_values = lines[i + 1].strip().split()
            
            # Combine both lines into a single row
            combined_values = main_values + additional_values
            
            # Convert combined values to float and add to the data list
            data.append(list(map(float, combined_values)))
        
        # Convert data to a NumPy array
        return np.array(data)
