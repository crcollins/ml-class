import os
import time
from functools import partial

import numpy

from sklearn import svm
from sklearn import dummy
from sklearn import neighbors
from sklearn import linear_model
from sklearn import tree

from utils import load_data, test_clf_kfold, cross_clf_kfold
import features
import clfs

import matplotlib.pyplot as plt
import clfplot


if __name__ == '__main__':
    methods = ('b3lyp', )#'cam', 'm06hf')
    base_paths = ('noopt', ) + tuple(os.path.join('opt', x) for x in methods)
    file_paths = [x + '.txt' for x in methods]

    start = time.time()
    names, geom_paths, properties, ends = load_data(base_paths, file_paths)

    tuned_decay = partial(features.get_decay_feature, power=2, factor=.75)
    tuned_decay.__name__ = "get_tuned_decay_feature"
    tuned_centered = partial(features.get_centered_decay_feature, power=.75, factor=.5, H=.75)
    tuned_centered.__name__ = "get_tuned_centered_feature"

    FEATURE_FUNCTIONS = [
        features.get_null_feature,
        features.get_binary_feature,
        features.get_flip_binary_feature,
        features.get_decay_feature,
        tuned_decay,
        tuned_centered,
        features.get_centered_decay_feature,
        features.get_signed_centered_decay_feature,
        # features.get_coulomb_feature,
        # features.get_pca_coulomb_feature,
    ]

    FEATURES = {}
    for function in FEATURE_FUNCTIONS:
        if function.__name__.startswith('get_'):
            key = function.__name__[4:]
        else:
            key = function.__name__
        temp = function(names, geom_paths)
        FEATURES[key] = numpy.concatenate((temp, ends), 1)

    PROPS = [numpy.matrix(x).T for x in properties]

    print "Took %.4f secs to load %d data points." % ((time.time() - start), PROPS[0].shape[0])
    print "Sizes of Feature Matrices"
    for name, feat in FEATURES.items():
        print "\t" + name, feat.shape
    print

    sets = (
        ('HOMO', PROPS[0]),
        ('LUMO', PROPS[1]),
        ('GAP', PROPS[2]),
    )

    CLFS = (
        ('Mean', dummy.DummyRegressor, {}),
        ('Linear', linear_model.LinearRegression, {}),
        ('LinearFix', clfs.LinearRegression, {}),
        ('LinearRidge', linear_model.Ridge, {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}),
        #('SVM', svm.SVR, {'C': [0.1, 1, 10, 100, 1000], 'gamma': [0.0001, 0.001, 0.01, 0.1]}),
        #('SVM Laplace', clfs.SVMLaplace, {'C': [0.1, 1, 10, 100, 1000], 'gamma': [0.0001, 0.001, 0.01, 0.1]}),
        #('k-NN', neighbors.KNeighborsRegressor, {'n_neighbors': [2, 3, 4, 5]}),
        #('Tree', tree.DecisionTreeRegressor, {'max_depth': [2, 3, 4, 5]}),
    )

    for NAME, PROP in sets:
        print NAME
        for FEAT_NAME, FEAT in FEATURES.items():
            print "\t" + FEAT_NAME

            #####for plot only####
            tempresult=numpy.zeros([len(CLFS),2])
            i=0
            #####for plot only####

            for CLF_NAME, CLF, KWARGS in CLFS:
                start = time.time()
                pair, (train, test) = cross_clf_kfold(FEAT, PROP, CLF, KWARGS, test_folds=5, cross_folds=10)
                finished = time.time() - start
                print "\t\t%s: %.4f +/- %.4f eV (%.4f secs)" % (CLF_NAME, test[0], test[1], finished), pair

                #####for plot only####
                tempresult[i] = [test[0], test[1]]
                i += 1
                #####for plot only####

            #####for plot only####
            clfplot.lineplot(tempresult[:, 0], tempresult[:, 1], [CLFS[i][0] for i in range(len(CLFS))], FEAT_NAME)
            #####for plot only####
            print
        #####for plot only####
        plt.show()
        #####for plot only####
        print
