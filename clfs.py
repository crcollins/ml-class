import numpy
from scipy.spatial.distance import cdist
from sklearn import svm

from utils import CLF


# Example Linear Regression
class LinearRegression(CLF):
    '''
    A basic implementation of Linear Regression.
    '''
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.weights = None

    def fit(self, X, y):
        X = numpy.matrix(X)
        y = numpy.matrix(y).T
        self.weights = numpy.linalg.pinv(X.T * X) * X.T * y

    def predict(self, X):
        return X * self.weights


def laplace_kernel_gen(sigma):
    def func(X, Y):
        return numpy.exp(sigma*-cdist(X,Y))
    return func

class SVMLaplace(svm.SVR):
    def __init__(self, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, tol=1e-3,
                 C=1.0, epsilon=0.1, shrinking=True, probability=False,
                 cache_size=200, verbose=False, max_iter=-1,
                 random_state=None):
        super(SVMLaplace, self).__init__(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, tol=tol,
                 C=C, epsilon=epsilon, shrinking=shrinking, probability=probability,
                 cache_size=cache_size, verbose=verbose, max_iter=max_iter,
                 random_state=random_state)
        self.kernel = laplace_kernel_gen(gamma)

def linear_sine_laplace_rbf_kernel_gen(sine_coef, lap_coef, rbf_coef, omega, lamda, sigma):
    def func(X, Y):
        distXY = cdist(X,Y)
        return numpy.dot(X, Y.T) + sine_coef*numpy.sin(omega*distXY) + lap_coef*numpy.exp(-lamda*distXY) + rbf_coef*numpy.exp( (-distXY**2) / (sigma*sigma) )
    return func

class SVM_Linear_Sine_Laplace_Rbf(svm.SVR):
    def __init__(self, kernel='rbf', degree=3, sine_coef=0.0, lap_coef=0.0, rbf_coef=0.0, omega=0.0, lamda=0.0, sigma=0.0, coef0=0.0, tol=1e-3,
                 C=1.0, epsilon=0.1, shrinking=True, probability=False,
                 cache_size=200, verbose=False, max_iter=-1,
                 random_state=None):
        super(SVM_Linear_Sine_Laplace_Rbf, self).__init__(kernel=kernel, degree=degree, gamma=0.0, coef0=coef0, tol=tol,
                 C=C, epsilon=epsilon, shrinking=shrinking, probability=probability,
                 cache_size=cache_size, verbose=verbose, max_iter=max_iter,
                 random_state=random_state)
        self.kernel = linear_sine_laplace_rbf_kernel_gen(sine_coef, lap_coef, rbf_coef, omega, lamda, sigma)

