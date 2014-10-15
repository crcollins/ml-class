import os
import time

import numpy

from sklearn import svm
from sklearn import dummy
from sklearn import neighbors
from sklearn import linear_model
from sklearn import tree

from utils import load_data, test_clf_kfold
import features
import clfs


if __name__ == '__main__':
    methods = ('b3lyp', )#'cam', 'm06hf')
    base_paths = ('noopt', ) + tuple(os.path.join('opt', x) for x in methods)
    file_paths = [x + '.txt' for x in methods]

    start = time.time()
    names, geom_paths, properties, ends = load_data(base_paths, file_paths)

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

    PROPS = [numpy.matrix(x).T for x in properties]

    print "Took %.4f secs to load %d data points." % ((time.time() - start), PROPS[0].shape[0])
    print "Sizes of Feature Matrices"
    for name, feat in FEATURES.items():
        print "\t" + name, feat.shape
    print

    sets = (
        ('HOMO', PROPS[0], 1, 0.1),
        ('LUMO', PROPS[1], 100, 0.01),
        ('GAP', PROPS[2], 1, 0.1),
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
