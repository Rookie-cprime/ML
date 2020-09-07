import numpy as np
class Perceptron(object):
    """
    Perceptron classifier
    Parameter
    ----------
    eta:float
        Learning rate(between 0.0 and 1.0)
    n_iter: int
        Passes over the training dataset
    random_state:Random number generator seed for random weight
    initialization
    Attibutes
    -----------------------------
    w_:ld-array
        Weights after fitting.
    errors_:list
        Number of misclassifications (updates) in each epoch.
    """
    def __init__(self,eta = 0.01,n_iter = 50,random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        selt.random_state = random_state
    def fit(self,X,y):
        """
        Fit training data.
        Parameter
        ------------------
        X: {array-like},shape = [n_sampels,n_features]
            Taining vectors,where n_sampels is the number of samples and
            n_features is the number of features.
        y: array-like,shape = [n_sampels]
            Target value
        Returns
        -------------------
        self:object
        """
        rgen = np.random.RandomState(self.random_state) #抓一组初始值
        self.w_ = rgen.normal(loc = 0.0,scale = 0.01,size=1+X.shape[1])#这里的+1指threshold见林轩田笔记
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi,target in zip(X,y):
                updata = self.eta*(target-self.predict(xi))
                self.w_[1:]+= updata*xi
                self.w_[0] += updata #默认x0 = 1
                errors += int(updata != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self,X):
        """Calculate net input"""
        return np.dot(X,self.w_[1:])+self.w_[0]
    def predict(self,X):
        """Return class label after unit step"""
        return np.where(self.net_input(X)>= 0.0,1,-1)


