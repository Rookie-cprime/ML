import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from AdalineGD import *
from percptron_plot import *

df = pd.read_csv('E:/GIT/ML/LGD/iris.data',header = None)
df.tail()
#select setosa and versicolor
y = df.iloc[0:100,4].values
y = np.where(y == "Iris-setosa",-1,1)
#extract sepal length and petal length
X = df.iloc[0:100,[0,2]].values

X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean())/X[:,0].std() #std standard mean everage
X_std[:,1] = (X[:,1] - X[:,1].mean())/X[:,1].std()


ada = AalineGD(n_iter = 15,eta = 0.01)
ada.fit(X_std[:,0:2],y)