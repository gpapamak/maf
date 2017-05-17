import numpy as np
import random
import theano


class DataStream:
    """Abstract class. Specifies the interface of a data stream.
    The user can request from the stream to generate a new data batch of a
    specified size. Useful for online learning."""

    def gen(self, N):
        """Generates a new data batch of size N."""
        raise NotImplementedError('This is an abstract method and should be overriden.')


class DataSubSampler(DataStream):
    """Given a data set, subsamples mini-batches from it."""

    def __init__(self, xs):

        # check that input is of the right type
        check = lambda t: isinstance(t, np.ndarray) and t.size and t.ndim
        assert isinstance(xs, list) and xs, 'Input must be a non-empty list.'
        assert check(xs[0]), 'Data must be given as real nonempty arrays.'
        N = xs[0].shape[0]
        for x in xs[1:]:
            assert check(x), 'Data must be given as real nonempty arrays.'
            Nk = x.shape[0]
            assert N == Nk, 'All data arrays must have the same number of elements in their first dimension.'

        # set remaining class properties
        self.index_stream = IndexSubSampler(N)
        self.xs = [ theano.shared(x.astype(theano.config.floatX), name='data'+str(i)) for i, x in enumerate(xs) ]

    def gen(self, N):
        """Generates a new data batch of size N from the data set."""

        assert isinstance(N, int) and N > 0, 'Batch size must be a positive integer.'

        n = self.index_stream.gen(N)
        return [x[n] for x in self.xs]


class IndexSubSampler(DataStream):
    """Subsamples minibatches of indices."""

    def __init__(self, num_idx):

        assert isinstance(num_idx, int) and num_idx > 0, 'Number of indices must be a positive integer.'

        self.num_idx = num_idx
        self.nn = range(num_idx)
        random.shuffle(self.nn)
        self.i = 0

    def gen(self, N):
        """Generates a new index batch of size N from 0:num_idx-1."""

        assert isinstance(N, int) and N > 0, 'Batch size must be a positive integer.'

        j = self.i + N
        times = j // self.num_idx
        new_i = j % self.num_idx
        n = []

        for t in xrange(times):
            n += self.nn[self.i:]
            random.shuffle(self.nn)
            self.i = 0

        n += self.nn[self.i:new_i]
        self.i = new_i

        return n


class IndexSubSamplerSeq(DataStream):
    """Subsamples minibatches of indices. Indices are sequentially grouped into minibatches."""

    def __init__(self, num_idx):

        assert isinstance(num_idx, int) and num_idx > 0, 'Number of indices must be a positive integer.'

        self.num_idx = num_idx
        self.nn = range(num_idx)
        self.i = 0

    def gen(self, N):
        """Generates a new index batch of size N from 0:num_idx-1."""

        assert isinstance(N, int) and N > 0, 'Batch size must be a positive integer.'

        j = self.i + N
        times = j // self.num_idx
        new_i = j % self.num_idx
        n = []

        for t in xrange(times):
            n += self.nn[self.i:]
            self.i = 0

        n += self.nn[self.i:new_i]
        self.i = new_i

        return n
