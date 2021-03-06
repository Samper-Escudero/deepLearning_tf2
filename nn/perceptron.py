import numpy as np

class Perceptron:
    def __init__(self,N, alpha=0.1):
        # initialize the weight matrix and store the learning rate
        self.W = np.random.randn(N +1) / np.sqrt(N)
        self.alpha = alpha

    def activation(self,x):
        # apply the activation function, which in this case is a step
        return 1 if x >0 else 0

    def fit(self,X,y,epochs = 10):
        # add the column for the bias parameters to ease training
        X = np.c_[X,np.ones((X.shape[0]))]
        for epoch in np.arange(0, epochs):
            for (x,target) in zip(X, y):
                p = self.activation(np.dot(x,self.W))
                if p != target:
                    error = p-target
                    self.W += -self.alpha*error*x

    def predict(self,X,addBias=True):
        X = np.atleast_2d(X)
        if addBias:
            X = np.c_[X,np.ones((X.shape[0]))]
        return self.activation(np.dot(X, self.W))
