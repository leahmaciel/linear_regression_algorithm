import math
import numpy as np
from linear_regression import *
from sklearn.datasets import make_regression
# Note: please don't add any new package, you should solve this problem using only the packages above.
#-------------------------------------------------------------------------
'''
    Problem 2: Apply your Linear Regression
    In this problem, use your linear regression method implemented in problem 1 to do the prediction.
    Play with parameters alpha and number of epoch to make sure your test loss is smaller than 1e-2.
    Report your parameter, your train_loss and test_loss 
    Note: please don't use any existing package for linear regression problem, use your own version.
'''

#--------------------------

n_samples = 200
X,y = make_regression(n_samples= n_samples, n_features=4, random_state=1)
y = np.array(y).T
X = np.array(X)
Xtrain, Ytrain, Xtest, Ytest = X[::2], y[::2], X[1::2], y[1::2]

#########################################
## INSERT YOUR CODE HERE

# first we need to make a feature matrix (Phi) for the training and testing data

n_features = 4 #sample data was made using 4 features so I selected 4 for the p input of compute_Phi

#initalize matrices to hold Phi for train and test
Phi_train = []
Phi_test = []

#loops through all the columns in X and computes Phi --> stacks the computed elements horizontally to keep the correct shape (n,p)
for i in range(Xtrain.shape[1]):
    Phi_column = compute_Phi(Xtrain[:, i], n_features)
    Phi_train.append(Phi_column)

Phi_train = np.hstack(Phi_train)

for i in range(Xtest.shape[1]):
    Phi_column = compute_Phi(Xtest[:, i], n_features)
    Phi_test.append(Phi_column)

Phi_test = np.hstack(Phi_test)



#set alpha and n_epoch values
alpha = 0.01
n_epoch = 5000

#train the model using the training sets
model = train(Phi_train, Ytrain, alpha= alpha, n_epoch= n_epoch)

#compute yhat and loss for training
yhat_train = compute_yhat(Phi_train, model)
training_loss = compute_L(yhat_train, Ytrain)
print(f"Training Loss: {training_loss:.4e}") #limit to four places using scientific notation
#Training Loss: 2.4723e-04


#compute yhat and loss for testing
yhat_test = compute_yhat(Phi_test, model)
testing_loss = compute_L(yhat_test, Ytest)
print(f"Testing Loss: {testing_loss:.4e}") #limit to four places using scientific notation
#Testing Loss: 2.4883e-03

#########################################

