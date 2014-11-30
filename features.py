import numpy
from numpy.linalg import norm

from sklearn import decomposition

from utils import tokenize, ARYL, RGROUPS, decay_function


# Example Feature function
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


def get_binary_feature(names, paths, limit=4):
    '''
    Creates a simple boolean feature vector based on whether or not a part is 
    in the name of the structure. 
    NOTE: This feature vector size scales O(N), where N is the limit.
    NOTE: Any parts of the name larger than the limit will be stripped off.

    >>> get_binary_feature(['4aa'], ['path/'], limit=1)
    matrix([[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])
    >>> get_binary_feature(['3'], ['path/'], limit=1)
    matrix([[0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]])
    >>> get_binary_feature(['4aa4aa'], ['path/'], limit=1)
    matrix([[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])
    >>> get_binary_feature(['4aa4aa'], ['path/'], limit=2)
    matrix([[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])
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


def get_flip_binary_feature(names, paths, limit=4):
    '''
    This creates a feature vector that is the same as the normal binary one
    with the addition of an additional element for each triplet to account
    for if the aryl group is flipped.
    NOTE: This feature vector size scales O(N), where N is the limit.
    NOTE: Any parts of the name larger than the limit will be stripped off.
    '''
    first = ARYL
    second = ['*'] + RGROUPS
    length = len(first) + 2 * len(second)

    vectors = []
    for name in names:
        features = []
        count = 0
        flips = []
        for token in tokenize(name):
            if token == '-':
                flips[-1] = 1
                continue

            base = second
            if token in first:
                if count == limit:
                    break
                count += 1
                flips.append(0)
                base = first
            temp = [0] * len(base)
            temp[base.index(token)] = 1
            features.extend(temp)

        # fill features to limit amount of groups
        features += [0] * length * (limit - count)
        flips += [0] * (limit - count)
        vectors.append(features + flips)

    return numpy.matrix(vectors)


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


def get_signed_centered_decay_feature(names, paths, power=1, H=1, factor=1):
    '''
    This feature vector works the same as the centered decay feature vector 
    with the addition that it takes into account the side of the center that
    the rings are on instead of just looking at the magnitude of the distance.
    '''
    first = ARYL
    second = ['*'] + RGROUPS
    length = len(first) + 2 * len(second)
    vector_map = first + 2 * second

    vectors = []
    for name in names:
        name = name.replace('-', '')  # no support for flipping yet
        
        end = tokenize(name)
        # One set is for the left (negative) side and the other is for the 
        # right side.
        partfeatures = [[0] * length, [0] * length]

        # Get the center index (x / 3 is to account for the triplet sets)
        # The x - 0.5 is to offset the value between index values.
        center = len(end) / 3. / 2. - 0.5
        for i, char in enumerate(end):
            # abs(x) is used to not make it not differentiate which 
            # direction each half of the structure is going relative to
            # the center
            count = (i / 3) - center
            # This is used as a switch to pick the correct side
            is_negative = count < 0
            count = abs(count)
            part = i % 3

            idx = vector_map.index(char)
            if char in second and part == 2:
                # If this is the second r group, change to use the second
                # R group location in the feature vector.
                idx = vector_map.index(char, idx + 1)

            # Needs to be optimized for power, H, and factor
            partfeatures[is_negative][idx] += decay_function(count + 1, power,
                                                            H, factor)
        vectors.append(partfeatures[0] + partfeatures[1])
    return numpy.matrix(vectors)


def get_coulomb_feature(names, paths):
    '''
    This feature vector is based on a distance matrix between all of the atoms
    in the structure with each element multiplied by the number of protons in 
    each of atom in the pair. The diagonal is 0.5 * protons ^ 2.4. The 
    exponent comes from a fit.
    This is based off the following work:
    M. Rupp, et al. Physical Review Letters, 108(5):058301, 2012.

    NOTE: This feature vector scales O(N^2) where N is the number of atoms in
    largest structure.
    '''
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


def get_pca_coulomb_feature(names, paths, dimensions=100):
    '''
    This feature vector takes the feature matrix from get_coulomb_feature and 
    does Principal Component Analysis on it to extract the N most influential
    dimensions. The goal of this is to reduce the size of the feature vector
    which can reduce overfitting, and most importantly dramatically reduce 
    running time.

    In principal, the number of dimensions used should correspond
    to at least 95% of the variability of the features (This is denoted by the
    `sum(pca.explained_variance_ratio_)`. For a full listing of the influence of 
    each dimension look at pca.explained_variance_ratio_.

    This method works by taking the N highest eigenvalues of the matrix (And
    their corresponding eigenvectors) and mapping the feature matrix into 
    this new lower dimensional space.
    '''
    feat = get_coulomb_feature(names, paths)
    pca = decomposition.PCA(n_components=dimensions)
    pca.fit(feat)
    # print pca.explained_variance_ratio_, sum(pca.explained_variance_ratio_)
    return numpy.matrix(pca.transform(feat))


def get_cndo_feature(names, paths)
    all_names = []
    properties = []
    valid_names = []
    valid_properties = []
    with open('./semi_feat/cndo.csv', 'r') as semi_feature_file:
        for line in semi_feature_file:
            temp = line.split(',')
            name, props = temp[1], temp[5:12]
            all_names.append(name)
            properties.append([float(x) for x in props])
    
    for validName in names:
        for i, dummyName in enumerate(all_names):
            if validName == dummyName:
                valid_names.append(dummyName)
                valid_properties.append(properties[i])
                break
    
    FEAT = numpy.zeros((len(valid_properties), 7))
    for i, x in enumerate(valid_properties):
        FEAT[i,:] = x
    return numpy.matrix(FEAT)


def get_indo_feature(names, paths)
    all_names = []
    properties = []
    valid_names = []
    valid_properties = []
    with open('./semi_feat/indo.csv', 'r') as semi_feature_file:
        for line in semi_feature_file:
            temp = line.split(',')
            name, props = temp[1], temp[5:12]
            all_names.append(name)
            properties.append([float(x) for x in props])
    
    for validName in names:
        for i, dummyName in enumerate(all_names):
            if validName == dummyName:
                valid_names.append(dummyName)
                valid_properties.append(properties[i])
                break
    
    FEAT = numpy.zeros((len(valid_properties), 7))
    for i, x in enumerate(valid_properties):
        FEAT[i,:] = x
    return numpy.matrix(FEAT)


def get_mndo_feature(names, paths)
    all_names = []
    properties = []
    valid_names = []
    valid_properties = []
    with open('./semi_feat/mndo.csv', 'r') as semi_feature_file:
        for line in semi_feature_file:
            temp = line.split(',')
            name, props = temp[1], temp[5:12]
            all_names.append(name)
            properties.append([float(x) for x in props])
    
    for validName in names:
        for i, dummyName in enumerate(all_names):
            if validName == dummyName:
                valid_names.append(dummyName)
                valid_properties.append(properties[i])
                break
    
    FEAT = numpy.zeros((len(valid_properties), 7))
    for i, x in enumerate(valid_properties):
        FEAT[i,:] = x
    return numpy.matrix(FEAT)

def get_am1_feature(names, paths)
    all_names = []
    properties = []
    valid_names = []
    valid_properties = []
    with open('./semi_feat/am1.csv', 'r') as semi_feature_file:
        for line in semi_feature_file:
            temp = line.split(',')
            name, props = temp[1], temp[5:12]
            all_names.append(name)
            properties.append([float(x) for x in props])
    
    for validName in names:
        for i, dummyName in enumerate(all_names):
            if validName == dummyName:
                valid_names.append(dummyName)
                valid_properties.append(properties[i])
                break
    
    FEAT = numpy.zeros((len(valid_properties), 7))
    for i, x in enumerate(valid_properties):
        FEAT[i,:] = x
    return numpy.matrix(FEAT)


def get_am1_feature(names, paths)
    all_names = []
    properties = []
    valid_names = []
    valid_properties = []
    with open('./semi_feat/pm3.csv', 'r') as semi_feature_file:
        for line in semi_feature_file:
            temp = line.split(',')
            name, props = temp[1], temp[5:12]
            all_names.append(name)
            properties.append([float(x) for x in props])
    
    for validName in names:
        for i, dummyName in enumerate(all_names):
            if validName == dummyName:
                valid_names.append(dummyName)
                valid_properties.append(properties[i])
                break
    
    FEAT = numpy.zeros((len(valid_properties), 7))
    for i, x in enumerate(valid_properties):
        FEAT[i,:] = x
    return numpy.matrix(FEAT)
