import numpy as np
import theano
import theano.tensor as tt

dtype = theano.config.floatX


class BatchNorm:

    def __init__(self, x, n_units, eps=1.0e-5):

        self.input = x

        # parameters
        self.log_gamma = theano.shared(np.zeros(n_units, dtype=dtype), name='log_gamma', borrow=True)
        self.beta = theano.shared(np.zeros(n_units, dtype=dtype), name='beta', borrow=True)
        self.parms = [self.log_gamma, self.beta]

        # minibatch statistics
        self.m = tt.mean(x, axis=0)
        self.v = tt.mean((x - self.m) ** 2, axis=0) + eps

        # transformation
        x_hat = (x - self.m) / tt.sqrt(self.v)
        self.y = tt.exp(self.log_gamma) * x_hat + self.beta

        # batch statistics to be used at test time
        self.bm = theano.shared(np.zeros(n_units, dtype=dtype), name='bm', borrow=True)
        self.bv = theano.shared(np.ones(n_units, dtype=dtype), name='bv', borrow=True)

        # theano evaluation functions, will be compiled when needed
        self.set_stats_f = None
        self.eval_f = None

    def set_batch_stats(self, x):
        """
        Sets the batch statistics to be equal to the statistics computed on dataset x.
        :param x: numpy array, rows are datapoints
        """

        if self.set_stats_f is None:
            self.set_stats_f = theano.function(
                inputs=[self.input],
                updates=[(self.bm, self.m), (self.bv, self.v)]
            )

        self.set_stats_f(x.astype(dtype))

    def eval(self, x):
        """
        Evaluates the batch norm transformation for input x.
        :param x: input as numpy array
        :return: output as numpy array
        """

        if self.eval_f is None:
            self.eval_f = theano.function(
                inputs=[self.input],
                outputs=[self.y],
                givens=[(self.m, self.bm), (self.v, self.bv)]
            )

        return self.eval_f(x.astype(dtype))

    def eval_inv(self, y):
        """
        Evaluates the inverse batch norm transformation for output y.
        NOTE: this calculation is done with numpy and not with theano.
        :param y: output as numpy array
        :return: input as numpy array
        """

        x_hat = (y - self.beta.get_value(borrow=True)) * np.exp(-self.log_gamma.get_value(borrow=True))
        x = np.sqrt(self.bv.get_value(borrow=True)) * x_hat + self.bm.get_value(borrow=True)

        return x
