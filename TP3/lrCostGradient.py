import numpy as np
from sigmoid import sigmoid

def h(theta, X):
    h = 0
    h = sigmoid(X@theta)
    return h

def lrCostGradient(theta, X, y, Lambda):
    """computes the gradient of the cost  w.r.t. to the parameters 
    theta for regularized logistic regression .
    """

    # préambule
    m,n = X.shape # m = 5; n = 4
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

    grad1 = ((h(theta, X) - y).T @ X)
    grad2 = Lambda*np.concatenate((np.zeros([1,1]),theta[1:]),axis=0).T
    grad = (grad1 + grad2)/m

    # =============================================================

    return grad.flatten() # ATTENTION: à conserver pour utiliser scipy.optimization.fmin_cg
