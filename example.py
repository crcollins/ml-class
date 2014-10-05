import re

from numpy.linalg import norm


ARYL = ['2', '3', '4', '11', '12']
ARYL0 = ['2', '3', '11']
RGROUPS = ['a', 'e', 'f', 'i', 'l']


def tokenize(string):
    '''
    Tokenizes a given string into the proper name segments. This includes the 
    addition of '*' tokens for aryl groups that do not support r groups.

    >>> tokenize('4al')
    ['4', 'a', 'l']
    >>> tokenize('4al12ff')
    ['4', 'a', 'l', '12', 'f', 'f']
    >>> tokenize('3')
    ['3', '*', '*']
    >>> tokenize('BAD')
    ValueError: Bad Substituent Name(s): ['BAD']
    '''

    match = '(1?\d|-|[%s])' % ''.join(RGROUPS)
    tokens = [x for x in re.split(match, string) if x]

    valid_tokens = set(ARYL + RGROUPS + ['-'])

    invalid_tokens = set(tokens).difference(valid_tokens)
    if invalid_tokens:
        raise ValueError("Bad Substituent Name(s): %s" % str(list(invalid_tokens)))

    new_tokens = []
    for token in tokens:
        new_tokens.append(token)
        if token in ARYL0:
            new_tokens.extend(['*', '*'])
    return new_tokens


def get_features(name, limit=4):
    '''
    Creates a simple boolean feature vector based on whether or not a part is 
    in the name of the structure.

    >>> get_features('4aa', limit=1)
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    >>> get_features('3', limit=1)
    [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    >>> get_features('4aa4aa', limit=1)
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    >>> get_features('4aa', limit=2)
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0]
    '''
    first = ARYL
    second = ['*'] + RGROUPS
    length = len(first) + 2 * len(second)

    features = []
    name = name.replace('-', '')  # no support for flipping yet
    count = 0
    for token in tokenize(name):
        base = second
        if token in first:
            if count == limit:
                break
            count += 1
            base = first
        temp = [0] * len(base)
        temp[base.index(token)] = 1
        features.extend(temp)

    # fill features to limit amount of groups
    features += [0] * length * (limit - count)
    return features


def get_features_coulomb(path):
    coords = []
    other = []
    types = {'C': 6, 'H': 1, 'O': 8}
    with open(path, 'r') as f:
        print path
        for line in f:
            ele, x, y, z = line.strip().split()
            point = (float(x), float(y), float(z))
            coords.append(numpy.matrix(point))
            other.append(types[ele])

    data = []
    for i, x in enumerate(coords):
        for j, y in enumerate(coords[:i + 1]):
            if i == j:
                val = 0.5 * other[i] ** 2.4
            else:
                val = (other[i]*other[j])/norm(x-y)
            data.append(val)
    return data


if __name__ == '__main__':
    import os
    import random

    import numpy

    from sklearn import svm
    from sklearn import neighbors
    from sklearn import linear_model
    from sklearn import cross_validation
    from sklearn.metrics import mean_absolute_error

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


    features = []
    features2 = []
    homos = []
    names = []
    lumos = []
    gaps = []

    for i, name in enumerate(('b3lyp.txt', )):# 'cam.txt', 'm06hf.txt')):
        path = os.path.join('data', 'opt', 'b3lyp', name)
        with open(path, 'r') as f:
            for line in f:
                name, homo, lumo, gap = line.split()
                names.append(name)
                feat = get_features(name)
                temp = [0, 0, 0]
                temp[i] = 1
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
            train = int(len(feat)*.9)
            X_train = FEAT[:train,:]
            Y_train = PROP[:train,:]
            X_test = FEAT[train:,:]
            Y_test = PROP[train:,:]

            w = numpy.linalg.pinv(X_train.T * X_train) * X_train.T * Y_train

            mean_pred = numpy.abs(Y_train.mean() - Y_test)
            lin_pred = numpy.abs(X_test * w - Y_test)
            print 'Mean', mean_pred.mean(), "+/-", lin_pred.std()
            print 'Linear', lin_pred.mean(), "+/-", lin_pred.std()
            print 'SVM', test_clf_kfold(FEAT, PROP, svm.SVR(C=C, gamma=gamma))[1]
            print 'k-NN', test_clf_kfold(FEAT, PROP, neighbors.KNeighborsRegressor(n_neighbors=5))[1]
            print 
