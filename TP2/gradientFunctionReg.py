import numpy as np
from sigmoid import sigmoid

def h(theta, X):
    h = 0
    h = sigmoid(X@theta)
    return h

def gradientFunctionReg(theta, X, y, Lambda):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """
    
    # Initialize some useful values
    m,n = X.shape   # number of training examples and parameters
    theta = theta.reshape((n,1)) # due to the use of fmin_tnc

    grad = 0.

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the gradient of a particular choice of theta.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    # =============================================================


    grad1 = ((h(theta, X) - y).T @ X)
    grad2 = Lambda*np.concatenate((np.zeros([1,1]),theta[1:]),axis=0).T
    grad = (grad1 + grad2)/m

    # =============================================================

    return grad