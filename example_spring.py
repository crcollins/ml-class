import os
import time
from functools import partial

import numpy

from sklearn import svm
from sklearn import dummy
from sklearn import neighbors
from sklearn import linear_model
from sklearn import tree

from utils import load_data, test_clf_kfold, cross_clf_kfold, OptimizedCLF
import features
import clfs


if __name__ == '__main__':
    methods = ('b3lyp',)# 'cam', 'm06hf')
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
        ('SVM Linear Sine RBF Laplace', clfs.SVM_Linear_Sine_Laplace_Rbf, {'C': 1, 'sine_coef': 1, 'lap_coef': 1, 'rbf_coef': 1, 'omega': 1, 'lamda': 1, 'sigma': 1}),
    )

    for NAME, PROP in sets:
        print NAME
        for FEAT_NAME, FEAT in FEATURES.items():
            print "\t" + FEAT_NAME
            for CLF_NAME, CLF, KWARGS in CLFS:
                start = time.time()
                optclf = OptimizedCLF(FEAT, PROP, CLF, KWARGS)
                newclf = optclf.get_optimized_clf()
                #~ pair, (train, test) = cross_clf_kfold(FEAT, PROP, CLF, KWARGS, test_folds=5, cross_folds=10)
                finished = time.time() - start
                #~ print "\t\t%s: %.4f +/- %.4f eV (%.4f secs)" % (CLF_NAME, test[0], test[1], finished), pair

            print 
        print
