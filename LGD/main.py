import numpy as np
class AdalineGD(object):
    """ADAptive LInear NEuron classifier.
    Parameters
    ----------
    eta: float
    Learning rate (between 0.0 and 1.0)
    n_iter : int
    Passes over the training dataset
    random_state:int
    Random number generator seed for random weight
    initialization.

    Attributes
    ------------
    w_: Id-array
    Weights after fitting
    cost_: list
    Sum-of-squares cost function value in each epoch
    """
    def __init__(self,eta = 0.01, n_iter = 50,random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    def fit(self,X,y):
        """

        :param X: {array_like},shape = {n_samples,n_features}
        Training vectors,where n_samples is the number of samples
        and n_features is the number of features.
        :param y: array-like,shape = [n_sample]
        Target value
        :return:
        self:object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0,scale = 0.01,size = 1+X.shape[1])#loc everage scale sdandard size
        
