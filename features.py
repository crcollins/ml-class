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


def decay_function(distance, power=1, H=1, factor=1):
    return (factor * (distance ** -H)) ** power


@feature_function
def get_decay_feature(names, paths, power=1, H=1, factor=1):
    '''
    This feature vector works about the same as the binary feature vector
    with the exception that it does not have O(N) scaling as the length of
    the molecule gains more rings. This is because it treats the 
    interaction between rings as some decay as they move further from the
    "start" of the structure (the start of the name).
    '''
    first = ARYL
    second = ['*'] + RGROUPS
    length = len(first) + 2 * len(second)
    vector_map = first + 2 * second

    vectors = []
    for name in names:

        name = name.replace('-', '')  # no support for flipping yet
        end = tokenize(name)
        temp = [0] * length
        for i, char in enumerate(end):
            # Use i / 3 because the tokens come in sets of 3 (Aryl, R1, R2)
            # Use i % 3 to get which part it is in the set (Aryl, R1, R2)
            count, part = divmod(i, 3)

            idx = vector_map.index(char)
            if char in second and part == 2:
                # If this is the second r group, change to use the second
                # R group location in the feature vector.
                idx = vector_map.index(char, idx + 1)

            # Needs to be optimized for power, H, and factor
            # count + 1 is used so that the first value will be 1, and 
            # subsequent values will have their respective scaling.
            temp[idx] += decay_function(count + 1, power, H, factor)
        vectors.append(temp)
    return numpy.matrix(vectors)


@feature_function
def get_centered_decay_feature(names, paths, power=1, H=1, factor=1):
    '''
    This feature vector takes the same approach as the decay feature vector
    with the addition that it does the decay from the center of the structure.
    '''
    first = ARYL
    second = ['*'] + RGROUPS
    length = len(first) + 2 * len(second)
    vector_map = first + 2 * second

    vectors = []
    for name in names:

        name = name.replace('-', '')  # no support for flipping yet
        
        end = tokenize(name)
        partfeatures = [0] * length

        # Get the center index (x / 3 is to account for the triplet sets)
        # The x - 0.5 is to offset the value between index values.
        center = len(end) / 3. / 2. - 0.5
        for i, char in enumerate(end):
            # abs(x) is used to not make it not differentiate which 
            # direction each half of the structure is going relative to
            # the center
            count = abs((i / 3) - center)
            part = i % 3

            idx = vector_map.index(char)
            if char in second and part == 2:
                # If this is the second r group, change to use the second
                # R group location in the feature vector.
                idx = vector_map.index(char, idx + 1)

            # Needs to be optimized for power, H, and factor
            partfeatures[idx] += decay_function(count + 1, power, H, factor)
        vectors.append(partfeatures)
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
