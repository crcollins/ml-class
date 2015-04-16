import re
import os
from itertools import product
from multiprocessing import Pool, cpu_count

import numpy

from sklearn import cross_validation
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing


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


ARYL = ['2', '3', '4', '6', '11', '12', '13']
ARYL0 = ['2', '3', '11']
RGROUPS = ['a', 'd', 'e', 'f', 'h', 'i', 'l']


def tokenize(string, explicit_flips=False):
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

    match = '(\(\d+\)|1?\d|-|[%s])' % ''.join(RGROUPS)
    tokens = [x for x in re.split(match, string) if x]

    valid_tokens = set(ARYL + RGROUPS + ['-'])

    invalid_tokens = set(tokens).difference(valid_tokens)
    if invalid_tokens:
        raise ValueError("Bad Substituent Name(s): %s" % str(list(invalid_tokens)))

    new_tokens = []
    flipped = False
    for i, token in enumerate(tokens):
        if explicit_flips and i and token in ARYL:
            new_tokens.append(flipped*"-")
            flipped = False
        elif token == "-":
            flipped = True

        if not explicit_flips or token != "-":
            new_tokens.append(token)
        if token in ARYL0:
            new_tokens.extend(['*', '*'])
    if explicit_flips:
        new_tokens.append(flipped*"-")
    return new_tokens


def decay_function(distance, power=1, H=1, factor=1):
    return (factor * (distance ** -H)) ** power


def gauss_decay_function(x, sigma=6):
    return numpy.exp(-(x / float(sigma)) ** 2)


def load_data(base_paths, file_paths, atom_sets):
    '''
    Load data from data sets and return lists of structure names, full paths
    to the geometry data, the properties, and the meta data.
    '''
    names = []
    geom_paths = []
    properties = []
    ends = []

    for j, base_path in enumerate(base_paths):
        for i, file_path in enumerate(file_paths):
            for m, atom_set in enumerate(atom_sets):
                path = os.path.join('mol_data', base_path, atom_set, file_path)
                with open(path, 'r') as f:
                    for line in f:
                        temp = line.split()
                        name, props = temp[0], temp[1:]
                        names.append(name)

                        geom_path = os.path.join('mol_data', base_path, 'geoms', 'out', name + '.out')
                        geom_paths.append(geom_path)

                        properties.append([float(x) for x in props])

                        # Add part to feature vector to account for the 4 different data sets.
                        base_part = [i == k for k, x in enumerate(base_paths)]
                        # Add part to feature vector to account for the 3 different methods.
                        method_part = [j == k for k, x in enumerate(file_paths)]
                        # Add part to feature vector to account for the addition of N.
                        atom_part = [m == k for k, x in enumerate(atom_sets)]
                        # Add bias feature
                        bias = [1]
                        ends.append(base_part + method_part + atom_part + bias)

    return names, geom_paths, zip(*properties), ends


def load_data_length(base_paths, file_paths, atom_sets, max_length=2):
    '''
    Load data from data sets and return lists of structure names, full paths
    to the geometry data, the properties, and the meta data.
    '''
    short_names = {}
    names = []
    name_map = []
    geom_paths = []
    properties = []
    ends = []

    for j, base_path in enumerate(base_paths):
        for i, file_path in enumerate(file_paths):
            for m, atom_set in enumerate(atom_sets):
                path = os.path.join('mol_data', base_path, atom_set, file_path)
                with open(path, 'r') as f:
                    for line in f:
                        temp = line.split()
                        name, props = temp[0], temp[1:]

                        tokens = tokenize(name, explicit_flips=True)

                        aryl_count = sum([1 for x in tokens if x in ARYL])
                        is_long = aryl_count > max_length

                        for num in xrange(1, max_length+1):
                            short_name = ''.join(tokens[:num*4])
                            if short_name not in short_names:
                                if not is_long:
                                    short_names[short_name] = [[len(names)], []]
                                else:
                                    short_names[short_name] = [None, []]
                            if is_long:
                                short_names[short_name][1].append(len(names))
                            elif short_names[short_name][0] is None:
                                short_names[short_name][0] = [len(names)]
                        names.append(name)

                        geom_path = os.path.join('mol_data', base_path, 'geoms', 'out', name + '.out')
                        geom_paths.append(geom_path)

                        properties.append([float(x) for x in props])

                        # Add part to feature vector to account for the 4 different data sets.
                        base_part = [i == k for k, x in enumerate(base_paths)]
                        # Add part to feature vector to account for the 3 different methods.
                        method_part = [j == k for k, x in enumerate(file_paths)]
                        # Add part to feature vector to account for the addition of N.
                        atom_part = [m == k for k, x in enumerate(atom_sets)]
                        # Add bias feature
                        bias = [1]
                        ends.append(base_part + method_part + atom_part + bias)

    return names, geom_paths, zip(*properties), ends, short_names


def _parallel_params(params):
    '''
    This is a helper function to run the parallel code. It contains the same
    code that the cross_clf_kfold had in the inner loop.
    '''
    param_names, group, clf_base, X_train, y_train, test_folds = params
    params = dict(zip(param_names, group))
    clf = clf_base(**params)

    X_use = numpy.matrix(X_train)
    y_use = numpy.matrix(y_train).T
    test_mean, test_std = test_clf_kfold(X_use, y_use, clf, folds=test_folds)
    return test_mean


def cross_clf_kfold(X, y, clf_base, params_sets, cross_folds=10, test_folds=10, parallel=False):
    '''
    This runs cross validation of a clf given a set of hyperparameters to
    test. It does this by splitting the data into testing and training data,
    and then it passes the training data into the test_clf_kfold function
    to get the error. The hyperparameter set that has the lowest test error is
    then returned from the function and its respective error.
    '''
    groups = {}
    param_names = params_sets.keys()

    n_sets = len(list(product(*params_sets.values())))
    cross = numpy.zeros((cross_folds, n_sets))

    # Calculate the cross validation errors for all of the parameter sets.
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
            data.append((param_names, group, clf_base, X_train, y_train, test_folds))

        if parallel:
            pool = Pool(processes=min(cpu_count(), len(data)))
            results = pool.map(_parallel_params, data)

            pool.close()
            pool.terminate()
            pool.join()
        else:
            results = map(_parallel_params, data)

        cross[i,:] = results

    # Get the set of parameters with the lowest cross validation error
    idx = numpy.argmin(cross.mean(0))
    for j, group in enumerate(product(*params_sets.values())):
        if j == idx:
            params = dict(zip(param_names, group))
            break

    # Get test error for set of params with lowest cross val error
    # The random_state used for the kfolds must be the same as the one used
    # before
    clf = clf_base(**params)
    return params, test_clf_kfold(X, y, clf, folds=cross_folds)


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
