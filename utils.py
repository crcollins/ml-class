import re
import os
from itertools import product
from multiprocessing import Pool, cpu_count

import numpy

from sklearn import cross_validation
from sklearn.metrics import mean_absolute_error


class CLF(object):
    def __init__(self, **kwargs):
        '''
        Self initialization code goes here.
        '''
        pass

    def fit(self, X, y):
        '''
        X is a (N_samples, N_features) array.
        y is a (N_samples, ) array.
        NOTE: These are arrays and NOT matrices. To do matrix-like operations
        on them you need to convert them to a matrix with 
        numpy.matrix(X) (or you can use numpy.dot(X, y), and etc).
        Note: This method does not return anything, it only stores state
        for later calls to self.predict()
        '''
        raise NotImplementedError

    def predict(self, X):
        '''
        X is a (N_samples, N_features) array.
        NOTE: This input is also an array and NOT a matrix.
        '''
        raise NotImplementedError


ARYL = ['2', '3', '4', '11', '12']
ARYL0 = ['2', '3', '11']
RGROUPS = ['a', 'e', 'f', 'i', 'l']


def tokenize(string):
    '''
    Tokenizes a given string into the proper name segments. This includes the 
    addition of '*' tokens for aryl groups that do not support r groups.

    >>> tokenize('4al')
    ['4', 'a', 'l']
    >>> tokenize('4al12ff')
    ['4', 'a', 'l', '12', 'f', 'f']
    >>> tokenize('3')
    ['3', '*', '*']
    >>> tokenize('BAD')
    ValueError: Bad Substituent Name(s): ['BAD']
    '''

    match = '(1?\d|-|[%s])' % ''.join(RGROUPS)
    tokens = [x for x in re.split(match, string) if x]

    valid_tokens = set(ARYL + RGROUPS + ['-'])

    invalid_tokens = set(tokens).difference(valid_tokens)
    if invalid_tokens:
        raise ValueError("Bad Substituent Name(s): %s" % str(list(invalid_tokens)))

    new_tokens = []
    for token in tokens:
        new_tokens.append(token)
        if token in ARYL0:
            new_tokens.extend(['*', '*'])
    return new_tokens


def decay_function(distance, power=1, H=1, factor=1):
    return (factor * (distance ** -H)) ** power


def gauss_decay_function(x, sigma=6):
    return numpy.exp(-(x / float(sigma)) ** 2)


def load_data(base_paths, file_paths):
    names = []
    geom_paths = []
    properties = []
    ends = []

    for j, base_path in enumerate(base_paths):
        for i, file_path in enumerate(file_paths):
            path = os.path.join('data', base_path, file_path)
            with open(path, 'r') as f:
                for line in f:
                    temp = line.split()
                    name, props = temp[0], temp[1:]
                    names.append(name)
                    
                    geom_path = os.path.join('data', base_path, 'geoms', 'out', name + '.out')
                    geom_paths.append(geom_path)

                    properties.append([float(x) for x in props])

                    # Add part to feature vector to account for the 4 different data sets.
                    base_part = [i == k for k, x in enumerate(base_paths)]
                    # Add part to feature vector to account for the 3 different methods.
                    method_part = [j == k for k, x in enumerate(file_paths)]
                    # Add bias feature
                    bias = [1]
                    ends.append(base_part + method_part + bias)

    return names, geom_paths, zip(*properties), ends


def _parallel_params(params):
    param_names, group, clf_base, X_train, y_train, X_test, y_test, test_folds = params
    params = dict(zip(param_names, group))
    clf = clf_base(**params)

    X_use = numpy.matrix(X_train)
    y_use = numpy.matrix(y_train).T
    (train_mean, train_std), (test_mean, test_std) = test_clf_kfold(X_use, y_use, clf, folds=test_folds)

    clf.fit(X_train, y_train)
    return mean_absolute_error(clf.predict(X_test), y_test)


def cross_clf_kfold(X, y, clf_base, params_sets, cross_folds=10, test_folds=10):
    groups = {}
    param_names = params_sets.keys()

    n_sets = len(list(product(*params_sets.values())))

    train = numpy.zeros((cross_folds, n_sets))
    test = numpy.zeros((cross_folds, n_sets))
    for i, (train_idx, test_idx) in enumerate(cross_validation.KFold(y.shape[0], 
                                                                    n_folds=cross_folds,
                                                                    shuffle=True,
                                                                    random_state=1)):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx].T.tolist()[0]
        y_test = y[test_idx].T.tolist()[0]

        data = []
        for j, group in enumerate(product(*params_sets.values())):
            data.append((param_names, group, clf_base, X_train, y_train, X_test, y_test, test_folds))

        pool = Pool(processes=cpu_count())
        results = pool.map(_parallel_params, data)

        pool.close()
        pool.terminate()
        pool.join()
        
        test[i,:] = results

    for j, group in enumerate(product(*params_sets.values())):
        groups[group] = (train.mean(0)[j], train.std(0)[j]), (test.mean(0)[j], test.std(0)[j])

    return sorted(groups.items(), key=lambda x:x[1][1][0])[0]


def test_clf_kfold(X, y, clf, folds=10):
    train = numpy.zeros(folds)
    test = numpy.zeros(folds)
    for i, (train_idx, test_idx) in enumerate(cross_validation.KFold(y.shape[0], 
                                                                    n_folds=folds,
                                                                    shuffle=True,
                                                                    random_state=1)):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx].T.tolist()[0]
        y_test = y[test_idx].T.tolist()[0]
        clf.fit(X_train, y_train)
        train[i] = mean_absolute_error(clf.predict(X_train), y_train)
        test[i] = mean_absolute_error(clf.predict(X_test), y_test)
    return (train.mean(), train.std()), (test.mean(), test.std())

