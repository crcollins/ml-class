import re

class CLF(object):
	def __init__(self, **kwargs):
		'''
		Self initialization code goes here.
		'''
		pass

	def fit(self, X, y):
		'''
		X is a (N_samples, N_features) array.
		y is a (N_samples, ) array.
		NOTE: These are arrays and NOT matrices. To do matrix-like operations
		on them you need to convert them to a matrix with 
		numpy.matrix(X) (or you can use numpy.dot(X, y), and etc).
		Note: This method does not return anything, it only stores state
		for later calls to self.predict()
		'''
		raise NotImplementedError

	def predict(self, X):
		'''
		X is a (N_samples, N_features) array.
		NOTE: This input is also an array and NOT a matrix.
		'''
		raise NotImplementedError

	@classmethod
	def clfs(cls):
		'''
		You do not need need to implement this. This is for book keeping.
		'''
		return cls.__subclasses__()


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


def decay_function(distance, power=1, H=1, factor=1):
    return (factor * (distance ** -H)) ** power