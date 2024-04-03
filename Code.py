#!/usr/bin/env python
# coding: utf-8

# In[68]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[69]:


# Load the dataset
data = pd.read_csv("garments_worker_productivity.csv")


# In[70]:


data.head()


# In[71]:


data.info()


# In[72]:


null_values = data.isnull().sum()
print("Null values in each column:")
print(null_values)


# In[73]:


data.fillna(0, inplace=True)


# In[74]:


threshold = 0.8
data['class_label'] = np.where(data['actual_productivity'] >= threshold, 1, 0)

# Determining the number of features for subplot layout
num_features = len(data.columns[:-2])  
num_rows = (num_features + 1) // 2  


# In[75]:


# Visualize the distribution of features and classes
plt.figure(figsize=(12, 10))
for i, column in enumerate(data.columns[:-2]):
    plt.subplot(num_rows, 2, i + 1)
    data[column].hist(bins=20)
    plt.title(column)
plt.subplot(num_rows, 2, num_rows*2)
data['class_label'].value_counts().plot(kind='bar', title='Class Distribution')
plt.show()


# In[76]:


# Normalize the data
scaler = StandardScaler()
data.iloc[:, 5:-1] = scaler.fit_transform(data.iloc[:, 5:-1])

# Spliting the data into training and test sets
X = data.iloc[:, 5:-1]  # Features
y = data['class_label']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Output shapes of the resulting sets
print("Training set shape:", X_train.shape, y_train.shape)
print("Test set shape:", X_test.shape, y_test.shape)


# In[55]:


import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.biases_input_hidden = np.zeros((1, self.hidden_size))
        
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.biases_hidden_output = np.zeros((1, self.output_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def feedforward(self, X):
        # Input layer to hidden layer
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.biases_input_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        
        # Hidden layer to output layer
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.biases_hidden_output
        self.output = self.sigmoid(self.output_input)
        
        return self.output
    
    def backward(self, X, y, output, learning_rate):
        # Calculate error and delta for output layer
        self.output_error = y - output
        self.output_delta = self.output_error * self.sigmoid_derivative(output)
        
        # Calculate error and delta for hidden layer
        self.hidden_error = self.output_delta.dot(self.weights_hidden_output.T)
        self.hidden_delta = self.hidden_error * self.sigmoid_derivative(self.hidden_output)
        
        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(self.output_delta) * learning_rate
        self.biases_hidden_output += np.sum(self.output_delta) * learning_rate
        
        self.weights_input_hidden += X.T.dot(self.hidden_delta) * learning_rate
        self.biases_input_hidden += np.sum(self.hidden_delta) * learning_rate
    
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.feedforward(X)
            self.backward(X, y, output, learning_rate)
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}: Loss {loss}")
    
    def predict(self, X):
        return self.feedforward(X)

# Example usage
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

input_size = X.shape[1]
hidden_size = 20
output_size = y.shape[1]

nn = NeuralNetwork(input_size, hidden_size, output_size)
nn.train(X, y, epochs=10000, learning_rate=0.1)

# Test the trained model
print("Predictions after training:")
print(nn.predict(X))


# In[67]:


import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initializing weights and biases
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.biases_input_hidden = np.zeros((1, self.hidden_size))
        
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.biases_hidden_output = np.zeros((1, self.output_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def feedforward(self, X):
  
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.biases_input_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        
        # Hidden layer to output layer
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.biases_hidden_output
        self.output = self.sigmoid(self.output_input)
        
        return self.output
    
    def predict(self, X):
        return self.feedforward(X)

# Initialize neural network with appropriate input size
input_size = X_test.shape[1]  
hidden_size = 20
output_size = 1  
nn = NeuralNetwork(input_size, hidden_size, output_size)

y_pred = nn.predict(X_test)

# Converting predicted probabilities to binary predictions
y_pred_binary = np.where(y_pred > 0.5, 1, 0)

# Accuracy
accuracy = accuracy_score(y_test, y_pred_binary)
print("Accuracy:", accuracy)

# Precision
precision = precision_score(y_test, y_pred_binary)
print("Precision:", precision)

# Recall
recall = recall_score(y_test, y_pred_binary)
print("Recall:", recall)

# F1-score
f1 = f1_score(y_test, y_pred_binary)
print("F1-score:", f1)

# ROC-AUC
roc_auc = roc_auc_score(y_test, y_pred)
print("ROC-AUC:", roc_auc)


# # Developing ANN using ML Libraries

# In[77]:


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data.fillna(0, inplace=True)

# Transforming actual_productivity to a discrete variable
threshold = 0.8
data['class_label'] = np.where(data['actual_productivity'] >= threshold, 1, 0)

# Normalize the data
scaler = StandardScaler()
data.iloc[:, 5:-1] = scaler.fit_transform(data.iloc[:, 5:-1])

# Spliting the data into training and test sets
X = data.iloc[:, 5:-1]  # Features
y = data['class_label']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construct the neural network model
mlp = MLPClassifier(hidden_layer_sizes=(20,), activation='relu', solver='adam', max_iter=1000)

# Train the neural network
mlp.fit(X_train, y_train)

# Test the neural network
y_pred = mlp.predict(X_test)

# Evaluating performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Performance metrics for neural network classifier using scikit-learn:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)


# # Parameter Tuning

# In[78]:


from sklearn.model_selection import GridSearchCV

# Define parameter grid for grid search
param_grid = {
    'hidden_layer_sizes': [(20,), (30,), (40,)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'max_iter': [1000, 1500, 2000]
}

# Initializing the grid search
grid_search = GridSearchCV(estimator=MLPClassifier(), param_grid=param_grid, cv=3, scoring='accuracy')

# Performing grid search to find the best parameters
grid_search.fit(X_train, y_train)

# best parameters found by grid search
best_params = grid_search.best_params_
print("Best parameters found by grid search:", best_params)

# Re-train the model with the best parameters
best_mlp = MLPClassifier(**best_params)
best_mlp.fit(X_train, y_train)

# Test the re-trained model
y_pred_tuned = best_mlp.predict(X_test)

accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
precision_tuned = precision_score(y_test, y_pred_tuned)
recall_tuned = recall_score(y_test, y_pred_tuned)
f1_tuned = f1_score(y_test, y_pred_tuned)

print("\nPerformance metrics for tuned neural network classifier:")
print("Accuracy:", accuracy_tuned)
print("Precision:", precision_tuned)
print("Recall:", recall_tuned)
print("F1-score:", f1_tuned)


# # Performing SVM machine learning

# In[79]:


from sklearn.svm import SVC

# SVM classifier
svm_classifier = SVC()

# Training the SVM classifier
svm_classifier.fit(X_train, y_train)

# Testing the SVM classifier
y_pred_svm = svm_classifier.predict(X_test)

# Evaluating performance
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)

print("Performance metrics for SVM classifier:")
print("Accuracy:", accuracy_svm)
print("Precision:", precision_svm)
print("Recall:", recall_svm)
print("F1-score:", f1_svm)


# # Comparing ANN with another SVM Model

# In[81]:


print("\nComparison of performance metrics:")

print("ANN Classifier:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

print("\nSVM Classifier:")
print("Accuracy:", accuracy_svm)
print("Precision:", precision_svm)
print("Recall:", recall_svm)
print("F1-score:", f1_svm)


# In[ ]:




