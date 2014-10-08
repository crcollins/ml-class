import os
import random

import numpy

from sklearn import svm
from sklearn import dummy
from sklearn import neighbors
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.metrics import mean_absolute_error

from features import get_features, get_features_coulomb


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
    features = []
    features2 = []
    homos = []
    names = []
    lumos = []
    gaps = []

    for j, name1 in enumerate(('noopt', 'opt/b3lyp', 'opt/cam', 'opt/m06hf')):
        for i, name in enumerate(('b3lyp.txt', 'cam.txt', 'm06hf.txt')):
            path = os.path.join('data', name1, name)
            with open(path, 'r') as f:
                for line in f:
                    name, homo, lumo, gap = line.split()
                    names.append(name)
                    feat = get_features(name)
                    # Add part to feature vector to account for the 3 different data sets.
                    temp = [0, 0, 0]
                    temp[i] = 1
                    feat += temp
                    temp = [0, 0, 0, 0]
                    temp[j] = 1
                    feat += temp
                    # Add bais feature
                    features.append(feat + [1])
                    homos.append(float(homo))
                    lumos.append(float(lumo))
                    gaps.append(float(gap))

    # for name in names:
    #     path = os.path.join('data', 'opt', 'b3lyp', 'geoms', name+'.out')
    #     features2.append(get_features_coulomb(path))

    temp = list(zip(features, homos, lumos, gaps))
    random.shuffle(temp)
    features, homos, lumos, gaps = zip(*temp)

    FEAT0 = numpy.matrix(features)
    HOMO = numpy.matrix(homos).T
    LUMO = numpy.matrix(lumos).T
    GAP = numpy.matrix(gaps).T
    # N = max(len(x) for x in features2)

    # FEAT2 = numpy.zeros((len(features2), N))
    # for i, x in enumerate(features2):
    #     for j, y in enumerate(x):
    #         FEAT2[i,j] = y
    # FEAT2 = numpy.matrix(FEAT2)

    sets = (
        ('HOMO', HOMO, 1, 0.1),
        ('LUMO', LUMO, 100, 0.01),
        ('GAP', GAP, 1, 0.1),
    )

    for NAME, PROP, C, gamma in sets:
        for FEAT in (FEAT0, ):# FEAT2):
            print NAME
            print 'Mean', "%.4f +/- %.4f eV" % test_clf_kfold(FEAT, PROP, dummy.DummyRegressor())[1]
            print 'Linear', "%.4f +/- %.4f eV" % test_clf_kfold(FEAT, PROP, linear_model.LinearRegression())[1]
            print 'Linear Ridge', "%.4f +/- %.4f eV" % test_clf_kfold(FEAT, PROP, linear_model.Ridge(alpha=1))[1]
            print 'SVM', "%.4f +/- %.4f eV" % test_clf_kfold(FEAT, PROP, svm.SVR(C=C, gamma=gamma))[1]
            print 'k-NN', "%.4f +/- %.4f eV" % test_clf_kfold(FEAT, PROP, neighbors.KNeighborsRegressor(n_neighbors=5))[1]
            print 
