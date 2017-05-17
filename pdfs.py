from itertools import izip

import numpy as np
import numpy.random as rng
import scipy.stats
import scipy.misc

import util


class Generator:
    """Abstract class that implements a data generator."""

    def gen(self, n_samples=1):
        raise NotImplementedError('Abstract method, should be implemented in a subclass.')

    def __call__(self, n_samples=1):
        return self.gen(n_samples)


class Gaussian(Generator):
    """Implements a gaussian pdf. Focus is on efficient multiplication, division and sampling."""

    def __init__(self, m=None, P=None, U=None, S=None, Pm=None):
        """
        Initialize a gaussian pdf given a valid combination of its parameters. Valid combinations are:
        m-P, m-U, m-S, Pm-P, Pm-U, Pm-S
        :param m: mean
        :param P: precision
        :param U: upper triangular precision factor such that U'U = P
        :param S: covariance
        :param Pm: precision times mean such that P*m = Pm
        """

        if m is not None:
            m = np.asarray(m)
            self.m = m
            self.ndim = m.size

            if P is not None:
                P = np.asarray(P)
                L = np.linalg.cholesky(P)
                self.P = P
                self.C = np.linalg.inv(L)
                self.S = np.dot(self.C.T, self.C)
                self.Pm = np.dot(P, m)
                self.logdetP = 2.0 * np.sum(np.log(np.diagonal(L)))

            elif U is not None:
                U = np.asarray(U)
                self.P = np.dot(U.T, U)
                self.C = np.linalg.inv(U.T)
                self.S = np.dot(self.C.T, self.C)
                self.Pm = np.dot(self.P, m)
                self.logdetP = 2.0 * np.sum(np.log(np.diagonal(U)))

            elif S is not None:
                S = np.asarray(S)
                self.P = np.linalg.inv(S)
                self.C = np.linalg.cholesky(S).T
                self.S = S
                self.Pm = np.dot(self.P, m)
                self.logdetP = -2.0 * np.sum(np.log(np.diagonal(self.C)))

            else:
                raise ValueError('Precision information missing.')

        elif Pm is not None:
            Pm = np.asarray(Pm)
            self.Pm = Pm
            self.ndim = Pm.size

            if P is not None:
                P = np.asarray(P)
                L = np.linalg.cholesky(P)
                self.P = P
                self.C = np.linalg.inv(L)
                self.S = np.dot(self.C.T, self.C)
                self.m = np.linalg.solve(P, Pm)
                self.logdetP = 2.0 * np.sum(np.log(np.diagonal(L)))

            elif U is not None:
                U = np.asarray(U)
                self.P = np.dot(U.T, U)
                self.C = np.linalg.inv(U.T)
                self.S = np.dot(self.C.T, self.C)
                self.m = np.linalg.solve(self.P, Pm)
                self.logdetP = 2.0 * np.sum(np.log(np.diagonal(U)))

            elif S is not None:
                S = np.asarray(S)
                self.P = np.linalg.inv(S)
                self.C = np.linalg.cholesky(S).T
                self.S = S
                self.m = np.dot(S, Pm)
                self.logdetP = -2.0 * np.sum(np.log(np.diagonal(self.C)))

            else:
                raise ValueError('Precision information missing.')

        else:
            raise ValueError('Mean information missing.')

    def gen(self, n_samples=1):
        """Returns independent samples from the gaussian."""

        z = rng.randn(n_samples, self.ndim)
        samples = np.dot(z, self.C) + self.m

        return samples

    def eval(self, x, ii=None, log=True):
        """
        Evaluates the gaussian pdf.
        :param x: rows are inputs to evaluate at
        :param ii: a list of indices specifying which marginal to evaluate. if None, the joint pdf is evaluated
        :param log: if True, the log pdf is evaluated
        :return: pdf or log pdf
        """

        if ii is None:
            xm = x - self.m
            lp = -np.sum(np.dot(xm, self.P) * xm, axis=1)
            lp += self.logdetP - self.ndim * np.log(2.0 * np.pi)
            lp *= 0.5

        else:
            m = self.m[ii]
            S = self.S[ii][:, ii]
            lp = scipy.stats.multivariate_normal.logpdf(x, m, S)
            lp = np.array([lp]) if x.shape[0] == 1 else lp

        res = lp if log else np.exp(lp)
        return res

    def __mul__(self, other):
        """Multiply with another gaussian."""

        assert isinstance(other, Gaussian)

        P = self.P + other.P
        Pm = self.Pm + other.Pm

        return Gaussian(P=P, Pm=Pm)

    def __imul__(self, other):
        """Incrementally multiply with another gaussian."""

        assert isinstance(other, Gaussian)

        res = self * other

        self.m = res.m
        self.P = res.P
        self.C = res.C
        self.S = res.S
        self.Pm = res.Pm
        self.logdetP = res.logdetP

        return res

    def __div__(self, other):
        """Divide by another gaussian. Note that the resulting gaussian might be improper."""

        assert isinstance(other, Gaussian)

        P = self.P - other.P
        Pm = self.Pm - other.Pm

        return Gaussian(P=P, Pm=Pm)

    def __idiv__(self, other):
        """Incrementally divide by another gaussian. Note that the resulting gaussian might be improper."""

        assert isinstance(other, Gaussian)

        res = self / other

        self.m = res.m
        self.P = res.P
        self.C = res.C
        self.S = res.S
        self.Pm = res.Pm
        self.logdetP = res.logdetP

        return res

    def __pow__(self, power, modulo=None):
        """Raise gaussian to a power and get another gaussian."""

        P = power * self.P
        Pm = power * self.Pm

        return Gaussian(P=P, Pm=Pm)

    def __ipow__(self, power):
        """Incrementally raise gaussian to a power."""

        res = self ** power

        self.m = res.m
        self.P = res.P
        self.C = res.C
        self.S = res.S
        self.Pm = res.Pm
        self.logdetP = res.logdetP

        return res

    def kl(self, other):
        """Calculates the kl divergence from this to another gaussian, i.e. KL(this | other)."""

        assert isinstance(other, Gaussian)
        assert self.ndim == other.ndim

        t1 = np.sum(other.P * self.S)

        m = other.m - self.m
        t2 = np.dot(m, np.dot(other.P, m))

        t3 = self.logdetP - other.logdetP

        t = 0.5 * (t1 + t2 + t3 - self.ndim)

        return t


class MoG(Generator):
    """Implements a mixture of gaussians."""

    def __init__(self, a, ms=None, Ps=None, Us=None, Ss=None, xs=None):
        """
        Creates a mog with a valid combination of parameters or an already given list of gaussian variables.
        :param a: mixing coefficients
        :param ms: means
        :param Ps: precisions
        :param Us: precision factors such that U'U = P
        :param Ss: covariances
        :param xs: list of gaussian variables
        """

        if ms is not None:

            if Ps is not None:
                self.xs = [Gaussian(m=m, P=P) for m, P in izip(ms, Ps)]

            elif Us is not None:
                self.xs = [Gaussian(m=m, U=U) for m, U in izip(ms, Us)]

            elif Ss is not None:
                self.xs = [Gaussian(m=m, S=S) for m, S in izip(ms, Ss)]

            else:
                raise ValueError('Precision information missing.')

        elif xs is not None:
            self.xs = xs

        else:
            raise ValueError('Mean information missing.')

        self.a = np.asarray(a)
        self.ndim = self.xs[0].ndim
        self.n_components = len(self.xs)

    def gen(self, n_samples=1, return_comps=False):
        """Generates independent samples from mog."""

        samples = np.empty([n_samples, self.ndim])
        ii = util.discrete_sample(self.a, n_samples)
        for i, x in enumerate(self.xs):
            idx = ii == i
            N = np.sum(idx.astype(int))
            samples[idx] = x.gen(N)

        return (samples, ii) if return_comps else samples

    def eval(self, x, ii=None, log=True):
        """
        Evaluates the mog pdf.
        :param x: rows are inputs to evaluate at
        :param ii: a list of indices specifying which marginal to evaluate. if None, the joint pdf is evaluated
        :param log: if True, the log pdf is evaluated
        :return: pdf or log pdf
        """

        ps = np.array([c.eval(x, ii, log) for c in self.xs]).T
        res = scipy.misc.logsumexp(ps + np.log(self.a), axis=1) if log else np.dot(ps, self.a)

        return res

    def __mul__(self, other):
        """Multiply with a single gaussian."""

        assert isinstance(other, Gaussian)

        ys = [x * other for x in self.xs]

        lcs = np.empty_like(self.a)

        for i, (x, y) in enumerate(izip(self.xs, ys)):

            lcs[i] = x.logdetP + other.logdetP - y.logdetP
            lcs[i] -= np.dot(x.m, np.dot(x.P, x.m)) + np.dot(other.m, np.dot(other.P, other.m)) - np.dot(y.m, np.dot(y.P, y.m))
            lcs[i] *= 0.5

        la = np.log(self.a) + lcs
        la -= scipy.misc.logsumexp(la)
        a = np.exp(la)

        return MoG(a=a, xs=ys)

    def __imul__(self, other):
        """Incrementally multiply with a single gaussian."""

        assert isinstance(other, Gaussian)

        res = self * other

        self.a = res.a
        self.xs = res.xs

        return res

    def __div__(self, other):
        """Divide by a single gaussian."""

        assert isinstance(other, Gaussian)

        ys = [x / other for x in self.xs]

        lcs = np.empty_like(self.a)

        for i, (x, y) in enumerate(izip(self.xs, ys)):

            lcs[i] = x.logdetP - other.logdetP - y.logdetP
            lcs[i] -= np.dot(x.m, np.dot(x.P, x.m)) - np.dot(other.m, np.dot(other.P, other.m)) - np.dot(y.m, np.dot(y.P, y.m))
            lcs[i] *= 0.5

        la = np.log(self.a) + lcs
        la -= scipy.misc.logsumexp(la)
        a = np.exp(la)

        return MoG(a=a, xs=ys)

    def __idiv__(self, other):
        """Incrementally divide by a single gaussian."""

        assert isinstance(other, Gaussian)

        res = self / other

        self.a = res.a
        self.xs = res.xs

        return res

    def calc_mean_and_cov(self):
        """Calculate the mean vector and the covariance matrix of the mog."""

        ms = [x.m for x in self.xs]
        m = np.dot(self.a, np.array(ms))

        msqs = [x.S + np.outer(mi, mi) for x, mi in izip(self.xs, ms)]
        S = np.sum(np.array([a * msq for a, msq in izip(self.a, msqs)]), axis=0) - np.outer(m, m)

        return m, S

    def project_to_gaussian(self):
        """Returns a gaussian with the same mean and precision as the mog."""

        m, S = self.calc_mean_and_cov()
        return Gaussian(m=m, S=S)

    def prune_negligible_components(self, threshold):
        """Removes all the components whose mixing coefficient is less than a threshold."""

        ii = np.nonzero((self.a < threshold).astype(int))[0]
        total_del_a = np.sum(self.a[ii])
        del_count = ii.size

        self.n_components -= del_count
        self.a = np.delete(self.a, ii)
        self.a += total_del_a / self.n_components
        self.xs = [x for i, x in enumerate(self.xs) if i not in ii]

    def kl(self, other, n_samples=10000):
        """Estimates the kl from this to another pdf, i.e. KL(this | other), using monte carlo."""

        x = self.gen(n_samples)
        lp = self.eval(x, log=True)
        lq = other.eval(x, log=True)
        t = lp - lq

        res = np.mean(t)
        err = np.std(t, ddof=1) / np.sqrt(n_samples)

        return res, err


def fit_gaussian(x, w=None):
    """Fits and returns a gaussian to a (possibly weighted) dataset using maximum likelihood."""

    if w is None:

        m = np.mean(x, axis=0)
        xm = x - m
        S = np.dot(xm.T, xm) / x.shape[0]

    else:
        m = np.dot(w, x)
        S = np.dot(x.T * w, x) - np.outer(m, m)

    return Gaussian(m=m, S=S)


def fit_mog(x, n_components, w=None, tol=1.0e-9, maxiter=float('inf'), verbose=False):
    """Fit and return a mixture of gaussians to (possibly weighted) data using expectation maximization."""

    x = x[:, np.newaxis] if x.ndim == 1 else x
    n_data, n_dim = x.shape

    # initialize
    a = np.ones(n_components) / n_components
    ms = rng.randn(n_components, n_dim)
    Ss = [np.eye(n_dim) for _ in xrange(n_components)]
    iter = 0

    # calculate log p(x,z), log p(x) and total log likelihood
    logPxz = np.array([scipy.stats.multivariate_normal.logpdf(x, ms[k], Ss[k]) for k in xrange(n_components)])
    logPxz += np.log(a)[:, np.newaxis]
    logPx = scipy.misc.logsumexp(logPxz, axis=0)
    loglik_prev = np.mean(logPx) if w is None else np.dot(w, logPx)

    while True:

        # e step
        z = np.exp(logPxz - logPx)

        # m step
        if w is None:
            Nk = np.sum(z, axis=1)
            a = Nk / n_data
            ms = np.dot(z, x) / Nk[:, np.newaxis]
            for k in xrange(n_components):
                xm = x - ms[k]
                Ss[k] = np.dot(xm.T * z[k], xm) / Nk[k]
        else:
            zw = z * w
            a = np.sum(zw, axis=1)
            ms = np.dot(zw, x) / a[:, np.newaxis]
            for k in xrange(n_components):
                xm = x - ms[k]
                Ss[k] = np.dot(xm.T * zw[k], xm) / a[k]

        # calculate log p(x,z), log p(x) and total log likelihood
        logPxz = np.array([scipy.stats.multivariate_normal.logpdf(x, ms[k], Ss[k]) for k in xrange(n_components)])
        logPxz += np.log(a)[:, np.newaxis]
        logPx = scipy.misc.logsumexp(logPxz, axis=0)
        loglik = np.mean(logPx) if w is None else np.dot(w, logPx)

        # check progress
        iter += 1
        diff = loglik - loglik_prev
        assert diff >= 0.0, 'Log likelihood decreased! There is a bug somewhere!'
        if verbose: print 'Iteration = {0}, log likelihood = {1}, diff = {2}'.format(iter, loglik, diff)
        if diff < tol or iter > maxiter: break
        loglik_prev = loglik

    return MoG(a=a, ms=ms, Ss=Ss)
