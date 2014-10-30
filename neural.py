import os
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

        # self.nn = self.build_network([n]+self.hidden_layers+[m])
        # self.nn = buildNetwork(n, 800, 50, m, bias=True, hiddenclass=SigmoidLayer)
        # self.nn = buildNetwork(n, 800, 35, m, bias=True, hiddenclass=SigmoidLayer)
        # self.nn = buildNetwork(n, 200, 100, 20, m, bias=True, hiddenclass=TanhLayer)
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


if __name__ == '__main__':
    methods = ('b3lyp', 'cam', 'm06hf')
    base_paths = tuple(os.path.join('opt', x) for x in methods)
    file_paths = [x + '.txt' for x in methods]# + ('indo_default', 'indo_b3lyp', 'indo_cam', 'indo_m06hf')]

    start = time.time()
    names, geom_paths, properties, ends = load_data(base_paths, file_paths)

    FEATURE_FUNCTIONS = [
        # features.get_null_feature,
        # features.get_binary_feature,
        # features.get_flip_binary_feature,
        # features.get_decay_feature,
        # tuned_decay,
        # tuned_centered,
        # features.get_centered_decay_feature,
        # features.get_signed_centered_decay_feature,
        # features.get_coulomb_feature,
        # features.get_pca_coulomb_feature,
        features.get_fingerprint_feature,
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

    X = numpy.array(FEATURES['fingerprint_feature'])
    y = numpy.array(numpy.concatenate(PROPS, 1))

    XTrain, yTrain, XTest, yTest = split_data(X, y)


    first = [25, 50, 100, 200, 400]
    second = [25, 50, 100, None]
    third = [10, 25, 50, None]

    clfs = {}
    temp = list(product(first, second, third))
    for layers in reversed(temp):
        n = X.shape[1]
        if len(y.shape) > 1:
            m = y.shape[1]
        else:
            m = 1

        layers = [n] + [x for x in layers if x] + [m]

        print layers
        clf = NeuralNet(layers)
        clfs[tuple(layers)] = clf
        clf.fit(XTrain, yTrain)
        clf.test_error.append(numpy.abs(clf.predict(XTest)-yTest).mean(0))
        print -1, clf.test_error[-1], numpy.linalg.norm(clf.test_error[-1])
        for i in xrange(100):
            clf.improve(10)
            clf.test_error.append(numpy.abs(clf.predict(XTest)-yTest).mean(0))
            print i, clf.test_error[-1], numpy.linalg.norm(clf.test_error[-1])

        print
