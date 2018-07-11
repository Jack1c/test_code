import numpy as np


class Perceptron(object):
    """
     参数:
     -----
     eta : float
     学习率 (在0.0到1.0直接)

     n_iter : int
     迭代次数

     属性:
     ----
     w_ : 1d-adrray
     权重

     errors_ : list
     错误
    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    def fit(self, X, y):
        """
        :param X:  {array-like} shape = [n_samples, n_features]
        :param y:  array-like, shape = [n_samples]
        :return:
        slef : object
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
