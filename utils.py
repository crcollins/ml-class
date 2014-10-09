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