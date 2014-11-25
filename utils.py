import re
import os
from itertools import product

import numpy
import scipy

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

    @classmethod
    def clfs(cls):
        '''
        You do not need need to implement this. This is for book keeping.
        '''
        return cls.__subclasses__()


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
                    
                    geom_path = os.path.join('data', base_path, 'geoms', name + '.out')
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
    test_mean, test_std = test_clf_kfold(X_use, y_use, clf, folds=test_folds)

    clf.fit(X_train, y_train)
    return mean_absolute_error(clf.predict(X_test), y_test)


def cross_clf_kfold(X, y, clf_base, params_sets, cross_folds=10, test_folds=10, parallel=False):
    groups = {}
    param_names = params_sets.keys()

    n_sets = len(list(product(*params_sets.values())))

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
        # This parallelization could probably be more efficient with an
        # iterator
        for group in product(*params_sets.values()):
            data.append((param_names, group, clf_base, X_train, y_train,
                        X_test, y_test, test_folds))

        if parallel:
            pool = Pool(processes=min(cpu_count(), len(data)))
            results = pool.map(_parallel_params, data)

            pool.close()
            pool.terminate()
            pool.join()
        else:
            results = map(_parallel_params, data)

        test[i,:] = results

    for j, group in enumerate(product(*params_sets.values())):
        groups[group] = (test.mean(0)[j], test.std(0)[j])

    # Sort groups.items() based on the test error and return the lowest one
    # x[1] is the value in groups [1] is the test values, and [0] is the mean
    return sorted(groups.items(), key=lambda x:x[1][0])[0]


def test_clf_kfold(X, y, clf, folds=10):
    results = numpy.zeros(folds)
    for i, (train_idx, test_idx) in enumerate(cross_validation.KFold(y.shape[0],
                                                                    n_folds=folds,
                                                                    shuffle=True,
                                                                    random_state=1)):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx].T.tolist()[0]
        y_test = y[test_idx].T.tolist()[0]
        clf.fit(X_train, y_train)
        results[i] = mean_absolute_error(clf.predict(X_test), y_test)
    return results.mean(), results.std()


###cross-validation for various CLF methods####
class OptimizedCLF(object):
    def __init__(self, X, y, func, params):
        self.params = params
        self.func = func
        self.X = X
        self.y = y
        self.optimized_clf = None
        self.optimized_params = None

    def errorfunction(self, *args):
        a = dict(zip(self.params.keys(), *args))
        clf = self.func(**a)
        train, test = test_clf_kfold(self.X, self.y, clf, folds=5)
        return test[0]

    def get_optimized_clf(self):
        if not len(self.params.keys()):
            self.optimized_clf = self.func()
        if self.optimized_clf is not None:
            return self.optimized_clf
        listparams = dict((k,v) for k,v in self.params.items() if type(v) in [list, tuple])
        itemparams = dict((k,v) for k,v in self.params.items() if type(v) not in [list, tuple])
        listvalues = []
        itemvalues = []
        #~ if listparams:
            #~ _, test = scan(self.X, self.y, self.func, listparams)
            #~ listvalues = []
            #~ temp = numpy.unravel_index(test.argmin(), test.shape)
            #~ for i, pick in enumerate(listparams.values()):
                #~ listvalues.append(pick[temp[i]])
            #~ listvalues = listvalues[::-1]
        if itemparams:
            bounds = ((1e-8, None), ) * len(self.params.keys())
            results = scipy.optimize.fmin_l_bfgs_b(
                self.errorfunction, self.params.values(),
                bounds=bounds,
                approx_grad=True, epsilon=1e-3)
            itemvalues = results[0].tolist()
        keys = listparams.keys() + itemparams.keys()
        values = listvalues + itemvalues
        self.optimized_params = dict(zip(keys, values))
        self.optimized_clf = self.func(**self.optimized_params)
        return self.optimized_clf
