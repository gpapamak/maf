import numpy as np
import numpy.random as rng
import theano
import theano.tensor as tt

import ml.models.neural_nets as nn
from ml.models.layers import BatchNorm
import util

dtype = theano.config.floatX


class CouplingLayer:
    """
    Coupling layer for Real NVP.
    """

    def __init__(self, x, mask, n_inputs, s_hiddens, s_act, t_hiddens, t_act):
        """
        Constructor of the backward computation graph.
        :param x: theano array, the input
        :param mask: theano array, a mask indicating which inputs are unchanged
        :param n_inputs: int, number of inputs
        :param s_hiddens: list of hidden widths for the scale net
        :param s_act: string, activation function for the scale net
        :param t_hiddens: list of hidden widths for the translate net
        :param t_act: string, activation function for the translate net
        """

        # save input arguments
        self.mask = mask
        self.n_inputs = n_inputs
        self.s_hiddens = s_hiddens
        self.s_act = s_act
        self.t_hiddens = t_hiddens
        self.t_act = t_act

        # masked input
        mx = mask * x

        # scale function
        self.s_net = nn.FeedforwardNet(n_inputs, mx)
        for h in s_hiddens:
            self.s_net.addLayer(h, s_act)
        self.s_net.addLayer(n_inputs, 'linear')

        # translate function
        self.t_net = nn.FeedforwardNet(n_inputs, mx)
        for h in t_hiddens:
            self.t_net.addLayer(h, t_act)
        self.t_net.addLayer(n_inputs, 'linear')

        # output
        s = self.s_net.output
        t = self.t_net.output
        self.u = mx + (1.0 - mask) * tt.exp(-s) * (x - t)

        # log det dx/dy
        self.logdet_dudx = -tt.sum((1.0 - mask) * s, axis=1)

        # parameters
        self.parms = self.s_net.parms + self.t_net.parms

        # theano evaluation function, will be compiled when first needed
        self.eval_forward_f = None

    def eval_forward(self, u):
        """
        Evaluates the layer forward, i.e. from random numbers u to output x.
        :param u: numpy array
        :return: numpy array
        """

        if self.eval_forward_f is None:

            # masked random numbers
            tt_u = tt.matrix('u')
            mu = self.mask * tt_u

            # scale net
            s_net = nn.FeedforwardNet(self.n_inputs, mu)
            for h in self.s_hiddens:
                s_net.addLayer(h, self.s_act)
            s_net.addLayer(self.n_inputs, 'linear')
            util.copy_model_parms(self.s_net, s_net)
            s = s_net.output

            # translate net
            t_net = nn.FeedforwardNet(self.n_inputs, mu)
            for h in self.t_hiddens:
                t_net.addLayer(h, self.t_act)
            t_net.addLayer(self.n_inputs, 'linear')
            util.copy_model_parms(self.t_net, t_net)
            t = t_net.output

            # transform u -> x
            x = mu + (1.0 - self.mask) * (tt_u * tt.exp(s) + t)

            # compile theano function
            self.eval_forward_f = theano.function(
                inputs=[tt_u],
                outputs=x
            )

        return self.eval_forward_f(u.astype(dtype))


class ConditionalCouplingLayer:
    """
    Coupling layer for the conditional version of Real NVP.
    """

    def __init__(self, x, y, mask, n_inputs, n_outputs, s_hiddens, s_act, t_hiddens, t_act):
        """
        Constructor of the backward computation graph.
        :param x: theano array, the conditional input
        :param y: theano array, the output
        :param mask: theano array, a mask indicating which outputs are unchanged
        :param n_inputs: int, number of conditional inputs
        :param n_outputs: int, number of outputs
        :param s_hiddens: list of hidden widths for the scale net
        :param s_act: string, activation function for the scale net
        :param t_hiddens: list of hidden widths for the translate net
        :param t_act: string, activation function for the translate net
        """

        # save input arguments
        self.mask = mask
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.s_hiddens = s_hiddens
        self.s_act = s_act
        self.t_hiddens = t_hiddens
        self.t_act = t_act

        # masked output
        my = mask * y

        # scale function
        self.s_net = nn.FeedforwardNet(n_inputs + n_outputs, tt.concatenate([x, my], axis=1))
        for h in s_hiddens:
            self.s_net.addLayer(h, s_act)
        self.s_net.addLayer(n_outputs, 'linear')

        # translate function
        self.t_net = nn.FeedforwardNet(n_inputs + n_outputs, tt.concatenate([x, my], axis=1))
        for h in t_hiddens:
            self.t_net.addLayer(h, t_act)
        self.t_net.addLayer(n_outputs, 'linear')

        # output
        s = self.s_net.output
        t = self.t_net.output
        self.u = my + (1.0 - mask) * tt.exp(-s) * (y - t)

        # log det dx/dy
        self.logdet_dudx = -tt.sum((1.0 - mask) * s, axis=1)

        # parameters
        self.parms = self.s_net.parms + self.t_net.parms

        # theano evaluation function, will be compiled when first needed
        self.eval_forward_f = None

    def eval_forward(self, x, u):
        """
        Evaluates the layer forward, i.e. from input x and random numbers u to output y.
        :param x: numpy array
        :param u: numpy array
        :return: numpy array
        """

        if self.eval_forward_f is None:

            # conditional input
            tt_x = tt.matrix('x')

            # masked random numbers
            tt_u = tt.matrix('u')
            mu = self.mask * tt_u

            # scale net
            s_net = nn.FeedforwardNet(self.n_inputs + self.n_outputs, tt.concatenate([tt_x, mu], axis=1))
            for h in self.s_hiddens:
                s_net.addLayer(h, self.s_act)
            s_net.addLayer(self.n_outputs, 'linear')
            util.copy_model_parms(self.s_net, s_net)
            s = s_net.output

            # translate net
            t_net = nn.FeedforwardNet(self.n_inputs + self.n_outputs, tt.concatenate([tt_x, mu], axis=1))
            for h in self.t_hiddens:
                t_net.addLayer(h, self.t_act)
            t_net.addLayer(self.n_outputs, 'linear')
            util.copy_model_parms(self.t_net, t_net)
            t = t_net.output

            # transform (x,u) -> y
            y = mu + (1.0 - self.mask) * (tt_u * tt.exp(s) + t)

            # compile theano function
            self.eval_forward_f = theano.function(
                inputs=[tt_x, tt_u],
                outputs=y
            )

        return self.eval_forward_f(x.astype(dtype), u.astype(dtype))


class RealNVP:
    """
    Real NVP, see Dinh et al, "Density estimation using Real NVP", 2016
    """

    def __init__(self, n_inputs, n_hiddens, s_act, t_act, n_layers, batch_norm=True):
        """
        Constructor.
        :param n_inputs: int, number of inputs
        :param n_hiddens: list of hidden widths for the nets in the coupling layers
        :param s_act: string, activation function for the scale net
        :param t_act: string, activation function for the translate net
        :param n_layers: int, number of coupling layers
        :param batch_norm: whether to use batch normalization between coupling layers
        :return:
        """

        # save input arguments
        self.n_inputs = n_inputs
        self.n_hiddens = n_hiddens
        self.s_act = s_act
        self.t_act = t_act
        self.n_layers = n_layers
        self.batch_norm = batch_norm

        # set up
        self.input = tt.matrix('x')
        self.u = self.input
        logdet_dudx = 0.0
        mask = theano.shared(np.arange(n_inputs, dtype=dtype) % 2, borrow=True)
        self.layers = []
        self.bns = []
        self.parms = []

        for _ in range(n_layers):

            # coupling layer
            layer = CouplingLayer(self.u, mask, n_inputs, n_hiddens, s_act, n_hiddens, t_act)
            mask = 1.0 - mask
            self.u = layer.u
            logdet_dudx += layer.logdet_dudx
            self.layers.append(layer)
            self.parms += layer.parms

            # batch normalization
            if batch_norm:
                bn = BatchNorm(self.u, n_inputs)
                self.u = bn.y
                logdet_dudx += tt.sum(bn.log_gamma) - 0.5 * tt.sum(tt.log(bn.v))
                self.parms += bn.parms
                self.bns.append(bn)

        # log likelihood
        self.L = -0.5 * n_inputs * np.log(2 * np.pi) - 0.5 * tt.sum(self.u ** 2, axis=1) + logdet_dudx
        self.trn_loss = -tt.mean(self.L)

        # theano evaluation functions, will be compiled when first needed
        self.eval_lprob_f = None
        self.eval_us_f = None

    def eval(self, x, log=True):
        """
        Evaluate log probabilities for given inputs.
        :param x: data matrix where rows are inputs
        :param log: whether to return probabilities in the log domain
        :return: list of log probabilities log p(x)
        """

        # compile theano function, if haven't already done so
        if self.eval_lprob_f is None:
            self.eval_lprob_f = theano.function(
                inputs=[self.input],
                outputs=self.L,
                givens=[(bn.m, bn.bm) for bn in self.bns] + [(bn.v, bn.bv) for bn in self.bns]
            )

        lprob = self.eval_lprob_f(x.astype(dtype))

        return lprob if log else np.exp(lprob)

    def gen(self, n_samples=1, u=None):
        """
        Generate samples.
        :param n_samples: number of samples
        :param u: random numbers to use in generating samples; if None, new random numbers are drawn
        :return: samples
        """

        x = rng.randn(n_samples, self.n_inputs).astype(dtype) if u is None else u

        if getattr(self, 'batch_norm', False):

            for layer, bn in zip(self.layers[::-1], self.bns[::-1]):
                x = bn.eval_inv(x)
                x = layer.eval_forward(x)

        else:

            for layer in self.layers[::-1]:
                x = layer.eval_forward(x)

        return x

    def calc_random_numbers(self, x):
        """
        Givan a dataset, calculate the random numbers real nvp uses internally to generate the dataset.
        :param x: numpy array, rows are datapoints
        :return: numpy array, rows are corresponding random numbers
        """

        # compile theano function, if haven't already done so
        if self.eval_us_f is None:
            self.eval_us_f = theano.function(
                inputs=[self.input],
                outputs=self.u
            )

        return self.eval_us_f(x.astype(dtype))


class ConditionalRealNVP:
    """
    Conditional version of Real NVP.
    """

    def __init__(self, n_inputs, n_outputs, n_hiddens, s_act, t_act, n_layers, batch_norm=True):
        """
        Constructor.
        :param n_inputs: int, number of inputs
        :param n_outputs: int, number of outputs
        :param n_hiddens: list of hidden widths for the nets in the coupling layers
        :param s_act: string, activation function for the scale net
        :param t_act: string, activation function for the translate net
        :param n_layers: int, number of coupling layers
        :param batch_norm: whether to use batch normalization between coupling layers
        :return:
        """

        # save input arguments
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hiddens = n_hiddens
        self.s_act = s_act
        self.t_act = t_act
        self.n_layers = n_layers
        self.batch_norm = batch_norm

        # set up
        self.input = tt.matrix('x')
        self.y = tt.matrix('y')
        self.u = self.y
        logdet_dudx = 0.0
        mask = theano.shared(np.arange(n_outputs, dtype=dtype) % 2, borrow=True)
        self.layers = []
        self.bns = []
        self.parms = []

        for _ in range(n_layers):

            # coupling layer
            layer = ConditionalCouplingLayer(self.input, self.u, mask, n_inputs, n_outputs, n_hiddens, s_act, n_hiddens, t_act)
            mask = 1.0 - mask
            self.u = layer.u
            logdet_dudx += layer.logdet_dudx
            self.layers.append(layer)
            self.parms += layer.parms

            # batch normalization
            if batch_norm:
                bn = BatchNorm(self.u, n_outputs)
                self.u = bn.y
                logdet_dudx += tt.sum(bn.log_gamma) - 0.5 * tt.sum(tt.log(bn.v))
                self.parms += bn.parms
                self.bns.append(bn)

        # log likelihood
        self.L = -0.5 * n_outputs * np.log(2 * np.pi) - 0.5 * tt.sum(self.u ** 2, axis=1) + logdet_dudx
        self.trn_loss = -tt.mean(self.L)

        # theano evaluation functions, will be compiled when first needed
        self.eval_lprob_f = None
        self.eval_us_f = None

    def eval(self, xy, log=True):
        """
        Evaluate log probabilities for given inputs.
        :param xy: (x, y) pair of numpy arrays, rows are datapoints
        :param log: whether to return probabilities in the log domain
        :return: list of log probabilities log p(y|x)
        """

        # compile theano function, if haven't already done so
        if self.eval_lprob_f is None:
            self.eval_lprob_f = theano.function(
                inputs=[self.input, self.y],
                outputs=self.L,
                givens=[(bn.m, bn.bm) for bn in self.bns] + [(bn.v, bn.bv) for bn in self.bns]
            )

        x, y = xy
        lprob = self.eval_lprob_f(x.astype(dtype), y.astype(dtype))

        return lprob if log else np.exp(lprob)

    def gen(self, x, n_samples=1, u=None):
        """
        Generate samples conditioned on x.
        :param x: numpy vector to condition on
        :param n_samples: number of samples
        :param u: random numbers to use in generating samples; if None, new random numbers are drawn
        :return: samples
        """

        xx = np.tile(x, [n_samples, 1])
        y = rng.randn(n_samples, self.n_outputs).astype(dtype) if u is None else u

        if getattr(self, 'batch_norm', False):

            for layer, bn in zip(self.layers[::-1], self.bns[::-1]):
                y = bn.eval_inv(y)
                y = layer.eval_forward(xx, y)

        else:

            for layer in self.layers[::-1]:
                y = layer.eval_forward(xx, y)

        return y

    def calc_random_numbers(self, xy):
        """
        Givan a dataset, calculate the random numbers real nvp uses internally to generate the dataset.
        :param xy: (x, y) pair of numpy arrays, rows are datapoints
        :return: numpy array, rows are corresponding random numbers
        """

        # compile theano function, if haven't already done so
        if self.eval_us_f is None:
            self.eval_us_f = theano.function(
                inputs=[self.input, self.y],
                outputs=self.u
            )

        x, y = xy
        return self.eval_us_f(x.astype(dtype), y.astype(dtype))
