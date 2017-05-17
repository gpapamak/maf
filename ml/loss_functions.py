import numpy as np
import theano.tensor as tt


def SquareError(x):
    """Square error loss function."""

    if x.ndim == 1:
        y = tt.vector('y')
        L = tt.mean((x - y) ** 2)

    elif x.ndim == 2:
        y = tt.matrix('y')
        L = tt.mean(tt.sum((x - y) ** 2, axis=1))

    else:
        raise ValueError('x must be either a vector or a matrix.')

    L.name = 'loss'

    return y, L


def CrossEntropy(x):
    """Cross entropy loss function. Only works for networks with one output."""

    if x.ndim == 1:
        pass

    elif x.ndim == 2:
        x = x[:, 0]

    else:
        raise ValueError('x must be either a vector or a matrix.')

    y = tt.vector('y')
    L = -tt.mean(y * tt.log(x) + (1-y) * tt.log(1-x))
    L.name = 'loss'

    return y, L


def MultiCrossEntropy(x):
    """Cross entropy loss function with multiple outputs."""

    assert x.ndim == 2, 'x must be a matrix.'

    y = tt.matrix('y')
    L = -tt.mean(tt.sum(y * tt.log(x), axis=1))
    L.name = 'loss'

    return y, L


def Accuracy(x):
    """Accuracy loss function. Mainly useful for validation."""

    if x.ndim == 1:
        pass

    elif x.ndim == 2:
        x = x.argmax(axis=1)

    else:
        raise ValueError('x must be either a vector or a matrix.')

    y = tt.vector('y')
    L = 100.0 * tt.mean(tt.eq(y, x))
    L.name = 'loss'

    return y, L


def WeightDecay(ws, wdecay):
    """Weight decay regularization."""

    assert wdecay > 0.0

    L = (wdecay / 2.0) * sum([tt.sum(w**2) for w in ws])
    return L


def SviRegularizer(mps, sps, wdecay):
    """
    The type of regularization that is used in stochastic variational inference. Here, we assume that the prior is
    a spherical zero-centred gaussian whose precision corresponds to the weight decay parameter.
    """

    assert wdecay > 0.0

    n_params = sum([mp.get_value().size for mp in mps])

    L1 = 0.5 * wdecay * (sum([tt.sum(mp**2) for mp in mps]) + sum([tt.sum(tt.exp(sp*2)) for sp in sps]))
    L2 = sum([tt.sum(sp) for sp in sps])
    Lc = 0.5 * n_params * (1.0 + np.log(wdecay))

    L = L1 - L2 - Lc

    return L
