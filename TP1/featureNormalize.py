import numpy as np


def featureNormalize(X):
    """
       returns a normalized version of X where
       the mean value of each feature is 0 and the standard deviation
       is 1. This is often a good preprocessing step to do when
       working with learning algorithms.
    """
    nbExample, nbFeature= X.shape
    X_norm =  X.copy()
    mu = np.zeros(nbFeature,)
    sigma = np.zeros(nbFeature,)
    for i in range(nbFeature):
        mu[i] = np.mean(X[:, i])
        sigma[i] = np.std(X[:, i])
    for i in range(nbFeature):
        for j in range(nbExample):
            X_norm[j, i] = (X_norm[j, i] - mu[i])/sigma[i]
    

    # ====================== YOUR CODE HERE ======================
    # Instructions: First, for each feature dimension, compute the mean
    #               of the feature and subtract it from the dataset,
    #               storing the mean value in mu. Next, compute the
    #               standard deviation of each feature and divide
    #               each feature by it's standard deviation, storing
    #               the standard deviation in sigma.
    #
    #               Note that X is a matrix where each column is a
    #               feature and each row is an example. You need
    #               to perform the normalization separately for
    #               each feature.
    #
    # Hint: You might find the 'mean' and 'std' functions useful.
    #
    





	# ============================================================

    return X_norm, mu, sigma
