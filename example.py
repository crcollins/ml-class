import os

import numpy

from sklearn import svm
from sklearn import dummy
from sklearn import neighbors
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.metrics import mean_absolute_error

from utils import FEATURE_FUNCTIONS
# UGH...
import features


def test_clf_kfold(X, y, clf, folds=10):
    train = numpy.zeros(folds)
    cross = numpy.zeros(folds)
    for i, (train_idx, test_idx) in enumerate(cross_validation.KFold(y.shape[0], n_folds=folds)):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx].T.tolist()[0]
        y_test = y[test_idx].T.tolist()[0]
        clf.fit(X_train, y_train)
        train[i] = mean_absolute_error(clf.predict(X_train), y_train)
        cross[i] = mean_absolute_error(clf.predict(X_test), y_test)
    return (train.mean(), train.std()), (cross.mean(), cross.std())


if __name__ == '__main__':
    feature_vectors = {}

    homos = []
    geom_paths = []
    lumos = []
    gaps = []

    methods = ('b3lyp', 'cam', 'm06hf')
    base_paths = ('noopt', ) + tuple(os.path.join('opt', x) for x in methods)
    file_paths = [x + '.txt' for x in methods]

    for j, base_path in enumerate(base_paths):
        for i, file_path in enumerate(file_paths):
            path = os.path.join('data', base_path, file_path)
            with open(path, 'r') as f:
                for line in f:
                    name, homo, lumo, gap = line.split()

                    homos.append(float(homo))
                    lumos.append(float(lumo))
                    gaps.append(float(gap))

                    geom_path = os.path.join('data', base_path, 'geoms', name + '.out')
                    geom_paths.append(geom_path)

                    for key, function in FEATURE_FUNCTIONS.items():
                        feat = function(name, geom_path)

                        # Add part to feature vector to account for the 4 different data sets.
                        base_part = [i == k for k, x in enumerate(base_paths)]

                        # Add part to feature vector to account for the 3 different methods.
                        method_part = [j == k for k, x in enumerate(file_paths)]

                        # Add bias feature
                        bias = [1]
                        
                        full = feat + base_part + method_part + bias
                        if key in feature_vectors:
                            feature_vectors[key].append(full)
                        else:
                            feature_vectors[key] = [full]


    FEATURES = {}
    for key, features in feature_vectors.items():
        lengths = set(len(x) for x in features)

        if len(lengths) > 1:
            # Hack to create feature matrix from hetero length feature vectors
            N = max(lengths)
            FEAT = numpy.zeros((len(features), N))
            
            for i, x in enumerate(features):
                for j, y in enumerate(x):
                    FEAT2[i,j] = y
            FEAT = numpy.matrix(FEAT)
        else:
            FEAT = numpy.matrix(features)

        FEATURES[key] = FEAT

    HOMO = numpy.matrix(homos).T
    LUMO = numpy.matrix(lumos).T
    GAP = numpy.matrix(gaps).T

    sets = (
        ('HOMO', HOMO, 1, 0.1),
        ('LUMO', LUMO, 100, 0.01),
        ('GAP', GAP, 1, 0.1),
    )

    CLFS = (
        ('Mean', dummy.DummyRegressor, {}),
        ('Linear', linear_model.LinearRegression, {}),
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

                train, test = test_clf_kfold(FEAT, PROP, CLF(**KWARGS))
                print "\t\t%s: %.4f +/- %.4f eV" % (CLF_NAME, test[0], test[1])
            print 
        print
