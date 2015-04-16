import os
import time
import sys

import numpy

from sklearn import svm
from sklearn import dummy
from sklearn import neighbors
from sklearn import linear_model
from sklearn import tree

from utils import load_data
import features
import clfs


def _multi_parallel_params(params):
    param_names, group, clf_base, X_train, y_train, test_folds = params
    params = dict(zip(param_names, group))
    clf = clf_base(**params)

    X_use = numpy.matrix(X_train)
    y_use = numpy.matrix(y_train).T
    test_mean, test_std = multi_test_clf_kfold(X_use, y_use, clf, folds=test_folds)
    return test_mean


def block_it(X, y, y_size, ends):
    X_block = numpy.tile(X, (y_size, 1))
    ns = X_block.shape[0]
    X_block = numpy.concatenate((X_block, ends), 1)
    y_block = numpy.concatenate(y[:,None], 1).flatten()
    return X_block, y_block


def multi_cross_clf_kfold(X, y, clf_base, params_sets, cross_folds=10, test_folds=10, parallel=False):
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
        y_train = y[train_idx,:]
        y_test = y[test_idx,:]

        data = []
        # This parallelization could probably be more efficient with an
        # iterator
        for group in product(*params_sets.values()):
            data.append((param_names, group, clf_base, X_train, y_train, test_folds))

        if parallel:
            pool = Pool(processes=min(cpu_count(), len(data)))
            results = pool.map(_multi_parallel_params, data)

            pool.close()
            pool.terminate()
            pool.join()
        else:
            results = map(_multi_parallel_params, data)

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
    return params, multi_test_clf_kfold(X, y, clf, folds=cross_folds)


def multi_test_clf_kfold(X, y, clf, folds=10):
    results = numpy.zeros((folds, y.shape[0]))

    ns = X.shape[0]
    ends = numpy.zeros((ns*y.shape[0], y.shape[0]))
    for i in xrange(y.shape[0]):
        ends[ns*i:ns*(i+1),i] = 1

    for i, (train_idx, test_idx) in enumerate(cross_validation.KFold(y.shape[1],
                                                                    n_folds=folds,
                                                                    shuffle=True,
                                                                    random_state=1)):
        numpy.set_printoptions(suppress=True)
        numpy.set_printoptions(precision=3)

        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[:,train_idx].T
        y_test = y[:,test_idx].T

        end_parts_train = []
        end_parts_test = []
        for j in xrange(y.shape[0]):
            base = (test_idx.shape[0] / y.shape[0]) * j
            end_parts_train.append(ends[train_idx+base,:])
            end_parts_test.append(ends[test_idx+base,:])
        end_train = numpy.concatenate(end_parts_train, 0)
        end_test = numpy.concatenate(end_parts_test, 0)
        scaler_y = preprocessing.StandardScaler().fit(y_train)


        # EEEEWWWWWWWWWW!!!!
        # transform modifies the original array
        y_trans_train = scaler_y.transform(y_train.copy())
        y_trans_test = scaler_y.transform(y_test.copy())

        X_block_train, y_block_train = block_it(X_train, y_trans_train, y.shape[0], end_train)
        X_block_test, y_block_test = block_it(X_test, y_trans_test, y.shape[0], end_test)

        clf.fit(X_block_train, y_block_train)
        pre = clf.predict(X_block_test)

        # import pdb; pdb.set_trace()
        pre2 = pre.reshape(*y_test.shape)

        results[i, :] = numpy.abs(scaler_y.inverse_transform(pre2) - y_test).mean(0)
    return results.mean(), results.std()


if __name__ == '__main__':
    # Change this to adjust which optimization sets to use
    methods = ('b3lyp', )#'cam', 'm06hf')

    # Change this to adjust the data sets to use
    base_paths = tuple(os.path.join('opt', x) for x in methods)# + ('noopt', )
    file_paths = [x + '.txt' for x in methods]
    atom_sets = ['O']#, 'N']

    start = time.time()
    names, geom_paths, properties, ends = load_data(base_paths, file_paths, atom_sets)

    # Change this to modify which feature vectors will be used for the testing
    FEATURE_FUNCTIONS = [
        # features.get_null_feature,
        # features.get_binary_feature,
        features.get_flip_binary_feature,
        # features.get_decay_feature,
        # features.get_gauss_decay_feature,
        # features.get_centered_decay_feature,
        # features.get_signed_centered_decay_feature,
        # features.get_coulomb_feature,
        # features.get_pca_coulomb_feature,
        # features.get_fingerprint_feature,
    ]
    PROPS = [numpy.matrix(x).T for x in properties]

    # Construct (name, vector) pairs to auto label features when iterating over them
    FEATURES = {}
    for function in FEATURE_FUNCTIONS:
        if function.__name__.startswith('get_'):
            key = function.__name__[4:]
        else:
            key = function.__name__
        temp = function(names, geom_paths)
        # Add the associtated file/data/opt meta data to each of the feature vectors
        FEATURES[key] = numpy.concatenate((temp, ends), 1)

    print "Took %.4f secs to load %d data points." % ((time.time() - start), PROPS[0].shape[0])
    print "Sizes of Feature Matrices"
    for name, feat in FEATURES.items():
        print "\t" + name, feat.shape
    print
    sys.stdout.flush()

    # Adjust the properties to test
    sets = (
	('ALL', numpy.concatenate(PROPS, 1)),
    )

    # Adjust the models/parameters to test
    # This is a collection of names, model pointers, and parameter values to cross validate
    # The parameters take the form of a dict with the keys being the parameter name for the model,
    # and the values being cross validated. For the cross validation parameters, it does a simple
    # cartesian product of all of the parameter lists and tests with all of them and then picks the
    # model with the lowest cross validation error and uses it for the final testing.
    CLFS = (
        ('Mean', dummy.DummyRegressor, {}),
#        ('Linear', linear_model.LinearRegression, {}),
#        ('LinearFix', clfs.LinearRegression, {}),
        ('LinearRidge', linear_model.Ridge, {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}),
        ('SVM', svm.SVR, {'C': [0.1, 1, 10, 100, 1000], 'gamma': [0.0001, 0.001, 0.01, 0.1]}),
#        ('SVM Laplace', clfs.SVMLaplace, {'C': [0.1, 1, 10, 100, 1000], 'gamma': [0.0001, 0.001, 0.01, 0.1]}),
#        ('k-NN', neighbors.KNeighborsRegressor, {'n_neighbors': [2, 3, 5, 8, 13]}),
#        ('Tree', tree.DecisionTreeRegressor, {'max_depth': [2, 3, 5, 8, 13, 21, 34, 55, 89]}),
    )

    results = {}
    for NAME, PROP in sets:
        print NAME
        results[NAME] = {}
        for FEAT_NAME, FEAT in FEATURES.items():
            print "\t" + FEAT_NAME
            results[NAME][FEAT_NAME] = {}
            for CLF_NAME, CLF, KWARGS in CLFS:
                start = time.time()

                import pdb; pdb.set_trace()
                pair, test = multi_cross_clf_kfold(FEAT, PROP, CLF, KWARGS, test_folds=5, cross_folds=2)
                finished = time.time() - start
                print "\t\t%s: %.4f +/- %.4f eV (%.4f secs)" % (CLF_NAME, test[0], test[1], finished), pair
                results[NAME][FEAT_NAME][CLF_NAME] = test[0]
            print
            sys.stdout.flush()
        print
