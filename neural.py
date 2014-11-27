import os
import sys
import time
import random
from itertools import product

import numpy

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, TanhLayer
from pybrain.structure import FullConnection

from utils import load_data
import features


class NeuralNet(object):
    def __init__(self, layers):
        self.layers = layers
        # self.hidden_layers = list(hidden_layers)
        self.ds = None
        self.train_error = []
        self.test_error = []
        self.test_error_norm = []

    def build_network(self, layers=None, end=1):
        layerobjects = []
        for item in layers:
            try:
                t, n = item
                if t == "sig":
                    if n == 0:
                        continue
                    layerobjects.append(SigmoidLayer(n))
            except TypeError:
                layerobjects.append(LinearLayer(item))

        n = FeedForwardNetwork()
        n.addInputModule(layerobjects[0])

        for i, layer in enumerate(layerobjects[1:-1]):
            n.addModule(layer)
            connection = FullConnection(layerobjects[i], layerobjects[i+1])
            n.addConnection(connection)

        n.addOutputModule(layerobjects[-1])
        connection = FullConnection(layerobjects[-2], layerobjects[-1])
        n.addConnection(connection)

        n.sortModules()
        return n

    def improve(self, n=10):
        trainer = BackpropTrainer(self.nn, self.ds)
        for i in xrange(n):
            self.train_error.append(trainer.train())
            # print self.train_error[-1]

    def fit(self, X, y):
        n = X.shape[1]
        if len(y.shape) > 1:
            m = y.shape[1]
        else:
            m = 1

        self.nn = buildNetwork(*self.layers, bias=True, hiddenclass=SigmoidLayer)

        ds = SupervisedDataSet(n, m)
        for i, row in enumerate(X):
            ds.addSample(row.tolist(), y[i])
        self.ds = ds
        self.improve()

    def predict(self, X):
        r = []
        for row in X:
            r.append(self.nn.activate(row))
        return numpy.array(r)


def split_data(X, y, percent=0.8):
    temp = range(len(X))
    random.shuffle(temp)
    X = X[temp,:]
    y = y[temp,:]

    split = int(percent*X.shape[0])
    XTrain = X[:split,:]
    XTest = X[split:,:]
    yTrain = y[:split]
    yTest = y[split:]
    return XTrain, yTrain, XTest, yTest


def test_architectures(X, y, layer_groups=None):
    if layer_groups is None:
        layer_groups = [
                        [25, 50, 100, 200, 400],
                        [25, 50, 100, None],
                        [10, 25, 50, None],
                        ]

    layers_set = set(tuple(y for y in x if y) for x in product(*layer_groups))
    XTrain, yTrain, XTest, yTest = split_data(X, y)

    n = X.shape[1]
    if len(y.shape) > 1:
        m = y.shape[1]
    else:
        m = 1

    clfs = {}
    for layers in layers_set:
        layers = (n, ) + layers + (m, )
        print layers

        clf = NeuralNet(layers)
        clfs[layers] = clf
        clf.fit(XTrain, yTrain)

        clf.test_error.append(numpy.abs(clf.predict(XTest)-yTest).mean(0))
        clf.test_error_norm.append(numpy.linalg.norm(clf.test_error[-1]))
        print -1, clf.test_error[-1], clf.test_error_norm[-1]
        for i in xrange(10):
            clf.improve(10)
            clf.test_error.append(numpy.abs(clf.predict(XTest)-yTest).mean(0))
            clf.test_error_norm.append(numpy.linalg.norm(clf.test_error[-1]))
            print i, clf.test_error[-1], clf.test_error_norm[-1]
        print
    return clfs


if __name__ == '__main__':
    methods = ('b3lyp', 'cam', 'm06hf')
    base_paths = tuple(os.path.join('opt', x) for x in methods)
    file_paths = [x + '.txt' for x in methods]# + ('indo_default', 'indo_b3lyp', 'indo_cam', 'indo_m06hf')]
    atom_sets = ['O', 'N']

    start = time.time()
    names, geom_paths, properties, ends = load_data(base_paths, file_paths, atom_sets)

    FEATURE_FUNCTIONS = [
        # features.get_null_feature,
        # features.get_binary_feature,
        features.get_flip_binary_feature,
        # features.get_decay_feature,
        # tuned_decay,
        # tuned_centered,
        # features.get_centered_decay_feature,
        # features.get_signed_centered_decay_feature,
        # features.get_coulomb_feature,
        # features.get_pca_coulomb_feature,
        # features.get_fingerprint_feature,
    ]

    FEATURES = {}
    for function in FEATURE_FUNCTIONS:
        if function.__name__.startswith('get_'):
            key = function.__name__[4:]
        else:
            key = function.__name__
        temp = function(names, geom_paths, size=1024)
        FEATURES[key] = numpy.concatenate((temp, ends), 1)

    PROPS = [numpy.matrix(x).T for x in properties]

    print "Took %.4f secs to load %d data points." % ((time.time() - start), PROPS[0].shape[0])
    print "Sizes of Feature Matrices"
    for name, feat in FEATURES.items():
        print "\t" + name, feat.shape
    print

    X = numpy.array(FEATURES['flip_binary_feature'])
    y = numpy.array(numpy.concatenate(PROPS, 1))

    XTrain, yTrain, XTest, yTest = split_data(X, y)

    n = XTrain.shape[1]
    if len(y.shape) > 1:
        m = y.shape[1]
    else:
        m = 1

    # layers = [n] + [25, 10] + [m]
    layers = [n] + [int(x) for x in sys.argv[1:]] + [m]

    print layers
    sys.stdout.flush()
    clf = NeuralNet(layers)
    clf.fit(XTrain, yTrain)
    clf.test_error.append(numpy.abs(clf.predict(XTest)-yTest).mean(0))
    print -1, clf.test_error[-1], numpy.linalg.norm(clf.test_error[-1])
    sys.stdout.flush()

    for i in xrange(5000):
        clf.improve(10)
        clf.test_error.append(numpy.abs(clf.predict(XTest)-yTest).mean(0))
        print i, clf.test_error[-1], numpy.linalg.norm(clf.test_error[-1])
        sys.stdout.flush()
