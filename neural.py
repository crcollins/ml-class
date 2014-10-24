import os
import time
import random

import numpy

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection

from utils import load_data
import features


class NeuralNet(object):
    def __init__(self, hidden_layers=None):
        self.hidden_layers = list(hidden_layers)
        self.ds = None

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
            print trainer.train()

    def fit(self, X, y):
        n = X.shape[1]
        if len(y.shape) > 1:
            m = y.shape[1]
        else:
            m = 1

        # self.nn = self.build_network([n]+self.hidden_layers+[m])
        # self.nn = buildNetwork(n, 800, 50, m, bias=True, hiddenclass=SigmoidLayer)
        self.nn = buildNetwork(n, 800, 35, m, bias=True, hiddenclass=SigmoidLayer)
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

    X = numpy.array(FEATURES['binary_feature'])
    y = numpy.array(numpy.concatenate(PROPS, 1))

    temp = range(len(X))
    random.shuffle(temp)
    X = X[temp,:]
    y = y[temp,:]

    split = int(.8*X.shape[0])
    XTrain = X[:split,:]
    XTest = X[split:,:]
    yTrain = y[:split]
    yTest = y[split:]

    clf = NeuralNet([('sig', 600), ('sig', 200)])
    clf.fit(XTrain, yTrain)
    temp = numpy.abs(clf.predict(XTest)-yTest).mean(0)
    print temp, numpy.linalg.norm(temp)
    for i in xrange(100):
        clf.improve(10)
        temp = numpy.abs(clf.predict(XTest)-yTest).mean(0)
        print temp, numpy.linalg.norm(temp)


