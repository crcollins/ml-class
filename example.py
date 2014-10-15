import os
import time

import numpy

from sklearn import svm
from sklearn import dummy
from sklearn import neighbors
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.metrics import mean_absolute_error

# UGH...
import features
import clfs


def test_clf_kfold(X, y, clf, folds=10):
    train = numpy.zeros(folds)
    test = numpy.zeros(folds)
    for i, (train_idx, test_idx) in enumerate(cross_validation.KFold(y.shape[0], n_folds=folds)):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx].T.tolist()[0]
        y_test = y[test_idx].T.tolist()[0]
        clf.fit(X_train, y_train)
        train[i] = mean_absolute_error(clf.predict(X_train), y_train)
        test[i] = mean_absolute_error(clf.predict(X_test), y_test)
    return (train.mean(), train.std()), (test.mean(), test.std())


if __name__ == '__main__':
    names = []
    geom_paths = []

    homos = []
    lumos = []
    gaps = []

    ends = []

    methods = ('b3lyp', )#'cam', 'm06hf')
    base_paths = ('noopt', ) + tuple(os.path.join('opt', x) for x in methods)
    file_paths = [x + '.txt' for x in methods]

    start = time.time()
    for j, base_path in enumerate(base_paths):
        for i, file_path in enumerate(file_paths):
            path = os.path.join('data', base_path, file_path)
            with open(path, 'r') as f:
                for line in f:
                    name, homo, lumo, gap = line.split()

                    names.append(name)
                    geom_path = os.path.join('data', base_path, 'geoms', name + '.out')
                    geom_paths.append(geom_path)

                    homos.append(float(homo))
                    lumos.append(float(lumo))
                    gaps.append(float(gap))

                    # Add part to feature vector to account for the 4 different data sets.
                    base_part = [i == k for k, x in enumerate(base_paths)]
                    # Add part to feature vector to account for the 3 different methods.
                    method_part = [j == k for k, x in enumerate(file_paths)]
                    # Add bias feature
                    bias = [1]

                    ends.append(base_part + method_part + bias)

    ENDS = numpy.matrix(ends)
    FEATURE_FUNCTIONS = [
        features.get_null_feature,
        features.get_binary_feature,
        features.get_flip_binary_feature,
        features.get_decay_feature,
        features.get_centered_decay_feature,
        features.get_signed_centered_decay_feature,
        # features.get_coulomb_feature,
        # features.get_pca_coulomb_feature,
    ]

    FEATURES = {}
    for function in FEATURE_FUNCTIONS:
        key = function.__name__.lstrip('get_')
        temp = function(names, geom_paths)
        FEATURES[key] = numpy.concatenate((temp, ends), 1)

    HOMO = numpy.matrix(homos).T
    LUMO = numpy.matrix(lumos).T
    GAP = numpy.matrix(gaps).T

    print "Took %.4f secs to load %d data points." % ((time.time() - start), HOMO.shape[0])
    print "Sizes of Feature Matrices"
    for name, feat in FEATURES.items():
        print "\t" + name, feat.shape
    print

    sets = (
        ('HOMO', HOMO, 1, 0.1),
        ('LUMO', LUMO, 100, 0.01),
        ('GAP', GAP, 1, 0.1),
    )

    CLFS = (
        ('Mean', dummy.DummyRegressor, {}),
        ('Linear', linear_model.LinearRegression, {}),
        ('LinearFix', clfs.LinearRegression, {}),
        ('LinearRidge', linear_model.Ridge, {'alpha': 1}),
        ('SVM', svm.SVR, {}),
        ('k-NN', neighbors.KNeighborsRegressor, {'n_neighbors': 2}),
    )


    for NAME, PROP, C, gamma in sets:
        print NAME
        for FEAT_NAME, FEAT in FEATURES.items():
            print "\t" + FEAT_NAME
            for CLF_NAME, CLF, KWARGS in CLFS:
                if CLF_NAME == 'SVM':
                    KWARGS = {'C': C, 'gamma': gamma}

                start = time.time()
                train, test = test_clf_kfold(FEAT, PROP, CLF(**KWARGS))
                finished = time.time() - start
                print "\t\t%s: %.4f +/- %.4f eV (%.4f secs)" % (CLF_NAME, test[0], test[1], finished)
            print 
        print
