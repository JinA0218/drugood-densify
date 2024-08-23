__all__ = ['str2bool', 'str2list', 'InfIterator']

def str2list(v):
    return v.split(',')

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

class InfIterator(object):
    def __init__(self, iterable):
        self.iterable = iterable
        self.iterator = iter(self.iterable)

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.iterable)
            return next(self.iterator)

    def __len__(self):
        return len(self.iterable)
