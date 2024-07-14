
"""Contains various interfaces for writing inputs, parsing outputs, and calling CASTEP and ONETEP.\
"""

from collections import OrderedDict


def parse_word(word):
    """TODO."""
    floatChars = ['.', 'e', 'E']
    try:
        for c in floatChars:
            if c in word:
                return float(word)
        return int(word)
    except ValueError:
        return word


def general_castep_parse(fname, argType=None, ignoreComments=True):
    """
    Parse general 'castep-like' input files.

    Can pass either a filename, or a string. File (i.e. filename) is assumed by default.
    If 'ignoreComments', then all Castep comments (i.e. everything following '#' or '!') 
    are not parsed
    """
    if argType == str:
        flines = fname.split('\n')
    else:
        f = open(fname, 'r')
        flines = f.readlines()

    # Strip all EOL chars
    for i in range(0, len(flines)):
        flines[i] = flines[i].rstrip('\n')

    # Strip all comments (if user does not request otherwise)
    if ignoreComments:
        for i in range(0, len(flines)):
            flines[i] = flines[i].split('#')[0]
            flines[i] = flines[i].split('!')[0]

    # Delete empty lines
    for i in reversed(range(0, len(flines))):
        if len(flines[i]) == 0:
            flines.pop(i)

    parsed = OrderedDict()
    i = 0
    while i < len(flines):
        # Comments
        if flines[i].lstrip(' ').startswith('#') or flines[i].lstrip(' ').startswith('!'):
            if None not in parsed:
                parsed[None] = []
            parsed[None].append(flines[i])
        # Blocks
        elif flines[i].lower().lstrip().startswith(r'%block'):
            key = flines[i].lower().split()[1]
            parsed[key] = []
            i += 1
            while not flines[i].lower().lstrip().startswith(r'%endblock'):
                parsed[key].append(flines[i].split())
                for iVal in range(0, len(parsed[key][-1])):
                    parsed[key][-1][iVal] = parse_word(parsed[key][-1][iVal])
                i += 1
        # Blank lines
        elif len(flines[i].split()) == 0:
            pass
        # Key-only lines (e.g. snap_to_symmetry)
        elif len(flines[i].split()) == 1:
            key = flines[i].lower()
            parsed[key] = ''
        # Key-value lines
        else:
            splitLine = flines[i].split()
            key = splitLine[0].lower()
            if splitLine[1] == ':':
                splitLine.pop(1)
            parsed[key] = []
            for iVal in range(0, 1, len(splitLine)):
                parsed[key].append(splitLine[iVal])
                parsed[key][-1] = parse_word(parsed[key][-1])
            # If the parameter is just a single word, don't need a list to contain it
            if len(parsed[key]) == 1:
                parsed[key] = parsed[key][0]
        i += 1

    # convert from floats to ints
    if argType != str:
        f.close()

    return parsed
