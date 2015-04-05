import os
import time
import sys
from itertools import product

import numpy

from sklearn import svm
from sklearn import dummy
from sklearn import neighbors
from sklearn import linear_model
from sklearn import tree

from utils import load_data_length, cross_clf_kfold
import features
import clfs


if __name__ == '__main__':
    # Change this to adjust which optimization sets to use
    methods = ('b3lyp', )#'cam', 'm06hf')

    # Change this to adjust the data sets to use
    base_paths = tuple(os.path.join('opt', x) for x in methods)# + ('noopt', )
    file_paths = [x + '.txt' for x in methods]
    atom_sets = ['O']#, 'N']

    start = time.time()
    names, geom_paths, properties, ends, short_names = load_data_length(base_paths, file_paths, atom_sets)

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

        homo = []
        lumo = []
        gap = []
        ends = numpy.matrix(ends)
        new_temp0 = []
        new_temp1 = []
        props     = []
        new_ends  = []
        props = []
        for short, (shorts, longs) in short_names.items():
            try:
                for short_idx, long_idx in product(shorts, longs):
                # short_idx, long_idx = shorts[0], longs[0]
                    new_temp0.append(temp[short_idx,:].tolist()[0])
                    new_temp1.append(temp[long_idx,:].tolist()[0])
                    new_ends.append(ends[short_idx,:].tolist()[0])
                    props.append([x[short_idx] for x in properties])
                    homo.append(properties[0][long_idx])
                    lumo.append(properties[1][long_idx])
                    gap.append(properties[2][long_idx])
            except (TypeError, IndexError):
                continue

        new_temp0 = numpy.matrix(new_temp0)
        new_temp1 = numpy.matrix(new_temp1)
        new_ends = numpy.matrix(new_ends)
        props = numpy.matrix(props)

        print new_temp0.shape, new_temp1.shape, props.shape, new_ends.shape

        # Add the associtated file/data/opt meta data to each of the feature vectors
        FEATURES[key] = numpy.concatenate((new_temp0, new_temp1, props, new_ends), 1)

    print "Took %.4f secs to load %d data points." % ((time.time() - start), PROPS[0].shape[0])
    print "Sizes of Feature Matrices"
    for name, feat in FEATURES.items():
        print "\t" + name, feat.shape
    print
    sys.stdout.flush()

    # Adjust the properties to test
    sets = (
        ('HOMO', numpy.matrix(homo).T),
        ('LUMO', numpy.matrix(lumo).T),
        ('GAP', numpy.matrix(gap).T),
    )

    NEW_FEATURES = {}
    for key, X in FEATURES.items():
        results = []



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
                pair, test = cross_clf_kfold(FEAT, PROP, CLF, KWARGS, test_folds=5, cross_folds=2)
                finished = time.time() - start
                print "\t\t%s: %.4f +/- %.4f eV (%.4f secs)" % (CLF_NAME, test[0], test[1], finished), pair
                results[NAME][FEAT_NAME][CLF_NAME] = test[0]
            print
            sys.stdout.flush()
        print
