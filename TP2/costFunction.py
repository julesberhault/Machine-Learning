import numpy as np
from sigmoid import sigmoid

def h(theta, X):
    h = 0
    h = sigmoid(X@theta)
    return h

def costFunction(theta, X, y):
    """ computes the cost of using theta as the
    parameter for logistic regression."""

	# Initialize some useful values
    m,n = X.shape   # number of training examples and parameters
    theta = theta.reshape((n,1)) # due to the use of fmin_tnc

    J = 0.
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #
    
    J = -y.T @ np.log(h(theta,X))-(np.ones((1,n))-y).T @ np.log(np.ones((1,n))-h(theta,X))
    J = (1/m)*J[0,0]
    
        
    # =============================================================
    
    return J

