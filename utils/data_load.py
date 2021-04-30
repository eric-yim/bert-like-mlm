def load(fpath):
    """
    Expects file where 
        each sequence is an individual line
        each line is a list of comma separated integers
        each line is of equal length
    Retunrs list of sequence lists
    """
    with open(fpath,'r') as f:
        lines = f.read().splitlines()
    return [to_int(line) for line in lines]
def to_int(line):
    return [int(a) for a in line.split(',')]