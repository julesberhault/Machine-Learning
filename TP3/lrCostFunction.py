import numpy as np
from sigmoid import sigmoid

def h(theta, X):
    h = 0
    h = sigmoid(X@theta)
    return h

def lrCostFunction(theta, X, y, Lambda):
    """computes the cost of using
    theta as the parameter for regularized logistic regression.
    """

    # preambule
    m,n = X.shape # 5,4
    theta = theta.reshape((n,1)) # (4,1)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #
    # Hint: The computation of the cost function and gradients can be
    #       efficiently vectorized. For example, consider the computation
    #
    #           sigmoid(X @ theta) or np.dot(X, theta)
    #
    #       Each row of the resulting matrix will contain the value of the
    #       prediction for that example. You can make use of this to vectorize
    #       the cost function and gradient computations. 
    #
 

    J1 = -y.T @ np.log(h(theta,X))-(np.ones((1,n))-y).T @ np.log(np.ones((1,n))-h(theta,X))
    J2 = theta.T @ theta
    J = (1/m)*(J1[0,0] + (Lambda/2)*J2)

    # =============================================================

    return J
