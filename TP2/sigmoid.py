import numpy as np

def sigmoid(z):
    """computes the sigmoid of z."""

    # initialisation
    g = 0.

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the sigmoid of each value of z (z can be a matrix,
    #               vector or scalar).
    
    if type(z) == 'int' or 'float':
        g=1./(1.+np.exp(-z))
            
    if type(z) == 'numpy.ndarray':
        m,n=z.shape
        g=np.zeros((m,n))
        for i in m:
            for j in n:
                g[i,j]=sigmoid(z[i,j])
    
    # =============================================================
    
    return  g
    
