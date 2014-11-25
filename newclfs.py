import numpy

from scipy.spatial.distance import cdist
from sklearn import svm

from utils import CLF





# def laplace_kernel_gen(sigma):
#     def func(X, Y):
#         return numpy.exp(sigma*-cdist(X,Y))
#     return func
#
# class SVMLaplace(svm.SVR):
#     def __init__(self, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, tol=1e-3,
#                  C=1.0, epsilon=0.1, shrinking=True, probability=False,
#                  cache_size=200, verbose=False, max_iter=-1,
#                  random_state=None):
#         super(SVMLaplace, self).__init__(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, tol=tol,
#                  C=C, epsilon=epsilon, shrinking=shrinking, probability=probability,
#                  cache_size=cache_size, verbose=verbose, max_iter=max_iter,
#                  random_state=random_state)
#         self.kernel = laplace_kernel_gen(gamma)

def laplace_rbf_kernel_gen(lap_coef, rbf_coef, lamda, sigma):
    def func(X, Y):
        distXY = cdist(X,Y)
        return lap_coef*numpy.exp(-lamda*distXY) + rbf_coef*numpy.exp((-distXY**2) / (sigma*sigma))
    return func

class SVM_Laplace_Rbf(svm.SVR):
    def __init__(self, kernel='rbf', degree=3, gamma=0.0, lap_coef=1.0, rbf_coef=1.0, lamda=1.0, sigma=1.0, coef0=0.0, tol=1e-3,
                 C=1.0, epsilon=0.1, shrinking=True, probability=False,
                 cache_size=200, verbose=False, max_iter=-1,
                 random_state=None):
        super(SVM_Laplace_Rbf, self).__init__(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, tol=tol,
                 C=C, epsilon=epsilon, shrinking=shrinking, probability=probability,
                 cache_size=cache_size, verbose=verbose, max_iter=max_iter,
                 random_state=random_state)
        self.kernel = laplace_rbf_kernel_gen(lap_coef, rbf_coef, lamda, sigma)





