import numpy
from numpy.linalg import norm

from utils import feature_function, tokenize, ARYL, RGROUPS


# Example Feature function
@feature_function
def get_null_feature(names, paths, **kwargs):
    '''
    names is a list of strings with the name of the structure (['4aa'])
    paths is a list of locations of the geometry files for that structures 
        (['data/noopt/geoms/4aa'])
    This function returns a single vector (1, N_features) in the form of a list 
    
    There is no need to add a bias term or try to split the structures based on
    which data set they came from, both of these will be handled as the data is
    loaded.

    NOTE: The '@FeatureFunction' at the start of this function is required.
    It is used to collect all the feature vectors together to reduce 
    duplication.
    '''
    return numpy.matrix(numpy.zeros((len(names), 0)))


@feature_function
def get_binary_feature(names, paths, limit=4):
    '''
    Creates a simple boolean feature vector based on whether or not a part is 
    in the name of the structure.

    >>> get_features(['4aa'], limit=1)
    [[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]
    >>> get_features(['3'], limit=1)
    [[0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]
    >>> get_features(['4aa4aa'], limit=1)
    [[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]
    >>> get_features(['4aa'], limit=2)
    [[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0]]
    '''
    first = ARYL
    second = ['*'] + RGROUPS
    length = len(first) + 2 * len(second)

    vectors = []
    for name in names:
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
        vectors.append(features)
    return numpy.matrix(vectors)


@feature_function
def get_coulomb_feature(names, paths):
    vectors = []
    for path in paths:
        coords = []
        other = []
        types = {'C': 6, 'H': 1, 'O': 8}
        with open(path, 'r') as f:
            # print path
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
        vectors.append(data)

    # Hack to create feature matrix from hetero length feature vectors
    N = max(len(x) for x in vectors)
    FEAT = numpy.zeros((len(vectors), N))
    for i, x in enumerate(vectors):
        for j, y in enumerate(x):
            FEAT[i,j] = y
    return numpy.matrix(FEAT)
