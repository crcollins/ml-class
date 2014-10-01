import re


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




if __name__ == '__main__':
    import os
    import random

    import numpy

    features = []
    homos = []
    lumos = []
    gaps = []

    path = os.path.join('data', 'noopt', 'b3lyp.txt')
    with open(path, 'r') as f:
        for line in f:
            name, homo, lumo, gap = line.split()
            feat = get_features(name)
            features.append(feat)
            homos.append(float(homo))
            lumos.append(float(lumo))
            gaps.append(float(gap))


    temp = list(zip(features, homos, lumos, gaps))
    random.shuffle(temp)
    features, homos, lumos, gaps = zip(*temp)

    FEAT = numpy.matrix(features)
    HOMO = numpy.matrix(homos).T
    LUMO = numpy.matrix(lumos).T
    GAP = numpy.matrix(gaps).T
    sets = (
        ('HOMO', HOMO),
        ('LUMO', LUMO),
        ('GAP', GAP),
    )

    for NAME, PROP in sets:
        print NAME
        train = int(len(feat)*.9)
        X_train = FEAT[:train,:]
        Y_train = PROP[:train,:]
        X_test = FEAT[train:,:]
        Y_test = PROP[train:,:]

        w = numpy.linalg.pinv(X_train.T * X_train) * X_train.T * Y_train

        mean_pred = numpy.abs(Y_train.mean() - Y_test)
        lin_pred = numpy.abs(X_test * w - Y_test)
        print 'Mean:', mean_pred.mean(), "+/-", mean_pred.std()
        print 'Linear:', lin_pred.mean(), "+/-", lin_pred.std()
        print