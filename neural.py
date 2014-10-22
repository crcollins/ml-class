import os
import time

import numpy

from utils import load_data
import features
import clfs


if __name__ == '__main__':
    methods = ('b3lyp', 'cam', 'm06hf')
    base_paths = tuple(os.path.join('opt', x) for x in methods)
    file_paths = [x + '.txt' for x in methods]# + ('indo_default', 'indo_b3lyp', 'indo_cam', 'indo_m06hf')]

    start = time.time()
    names, geom_paths, properties, ends = load_data(base_paths, file_paths)

    FEATURE_FUNCTIONS = [
        features.get_null_feature,
        features.get_binary_feature,
        # features.get_flip_binary_feature,
        features.get_decay_feature,
        # tuned_decay,
        # tuned_centered,
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

    X = numpy.array(FEATURES['binary_feature'])
    y = numpy.array(PROPS[0].T.tolist()[0])
    import random
    temp = range(len(X))
    random.shuffle(temp)
    X = X[temp,:]
    y = y[temp,:]

    split = int(.8*X.shape[0])
    XTrain = X[:split,:]
    XTest = X[split:,:]
    yTrain = y[:split]
    yTest = y[split:]

    clf = clfs.NeuralNet([('sig', 600), ('sig', 200)])
    clf.fit(XTrain, yTrain)
    print numpy.abs(clf.predict(XTest)-yTest).mean()
    for i in xrange(100):
        clf.improve(50)
        print numpy.abs(clf.predict(XTest)-yTest).mean()
