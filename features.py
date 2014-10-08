import re

import numpy
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