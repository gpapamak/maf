# train density estimators on various datasets

from __future__ import division

import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

import ml.trainers as trainers
import ml.models.mades as mades
import ml.models.mafs as mafs
import ml.models.nvps as nvps
import ml.step_strategies as ss
import ml.loss_functions as lf
import util
import datasets
import pdfs


# set paths
root_output = 'output/'   # where to save trained models
root_data = 'data/'       # where the datasets are

# holders for the datasets
data = None
data_name = None

# parameters for training
minibatch = 100
patience = 30
monitor_every = 1
weight_decay_rate = 1.0e-6
a_made = 1.0e-3
a_flow = 1.0e-4


def load_data(name):
    """
    Loads the dataset. Has to be called before anything else.
    :param name: string, the dataset's name
    """

    assert isinstance(name, str), 'Name must be a string'
    datasets.root = root_data
    global data, data_name

    if data_name == name:
        return

    if name == 'mnist':
        data = datasets.MNIST(logit=True, dequantize=True)
        data_name = name

    elif name == 'bsds300':
        data = datasets.BSDS300()
        data_name = name

    elif name == 'cifar10':
        data = datasets.CIFAR10(logit=True, flip=True, dequantize=True)
        data_name = name

    elif name == 'power':
        data = datasets.POWER()
        data_name = name

    elif name == 'gas':
        data = datasets.GAS()
        data_name = name

    elif name == 'hepmass':
        data = datasets.HEPMASS()
        data_name = name

    elif name == 'miniboone':
        data = datasets.MINIBOONE()
        data_name = name

    else:
        raise ValueError('Unknown dataset')


def is_data_loaded():
    """
    Checks whether a dataset has been loaded.
    :return: boolean
    """
    return data_name is not None


def create_model_id(model_name, mode, n_hiddens, act_fun, n_comps, batch_norm):
    """
    Creates an identifier for the provided model description.
    """

    assert is_data_loaded(), 'Dataset hasn\'t been loaded'

    delim = '_'
    id = data_name + delim + model_name + delim

    if mode is not None:
        if mode == 'sequential':
            id += 'seq' + delim
        elif mode == 'random':
            id += 'rnd' + delim
        else:
            raise ValueError('invalid mode')

    if batch_norm:
        id += 'bn' + delim

    for h in n_hiddens:
        id += str(h) + delim

    if n_comps is not None:
        id += 'layers' + delim + str(n_comps) + delim

    id += act_fun

    return id


def save_model(model, model_name, mode, n_hiddens, act_fun, n_comps, batch_norm):

    assert is_data_loaded(), 'Dataset hasn\'t been loaded'

    savedir = root_output + data_name + '/'
    util.make_folder(savedir)
    filename = create_model_id(model_name, mode, n_hiddens, act_fun, n_comps, batch_norm)

    util.save(model, savedir + filename + '.pkl')


def load_model(model_name, mode, n_hiddens, act_fun, n_comps=None, batch_norm=False):

    assert is_data_loaded(), 'Dataset hasn\'t been loaded'

    savedir = root_output + data_name + '/'
    filename = create_model_id(model_name, mode, n_hiddens, act_fun, n_comps, batch_norm)

    return util.load(savedir + filename + '.pkl')


def train(model, a):

    assert is_data_loaded(), 'Dataset hasn\'t been loaded'

    regularizer = lf.WeightDecay(model.parms, weight_decay_rate)

    trainer = trainers.SGD(
        model=model,
        trn_data=[data.trn.x],
        trn_loss=model.trn_loss + regularizer,
        val_data=[data.val.x],
        val_loss=model.trn_loss,
        step=ss.Adam(a=a)
    )

    trainer.train(
        minibatch=minibatch,
        patience=patience,
        monitor_every=monitor_every
    )


def train_cond(model, a):

    assert is_data_loaded(), 'Dataset hasn\'t been loaded'

    regularizer = lf.WeightDecay(model.parms, weight_decay_rate)

    trainer = trainers.SGD(
        model=model,
        trn_data=[data.trn.y, data.trn.x],
        trn_target=model.y,
        trn_loss=model.trn_loss + regularizer,
        val_data=[data.val.y, data.val.x],
        val_target=model.y,
        val_loss=model.trn_loss,
        step=ss.Adam(a=a)
    )

    trainer.train(
        minibatch=minibatch,
        patience=patience,
        monitor_every=monitor_every
    )


def train_made(n_hiddens, act_fun, mode):

    assert is_data_loaded(), 'Dataset hasn\'t been loaded'
    model = mades.GaussianMade(data.n_dims, n_hiddens, act_fun, mode=mode)
    train(model, a_made)
    save_model(model, 'made', mode, n_hiddens, act_fun, None, False)


def train_made_cond(n_hiddens, act_fun, mode):

    assert is_data_loaded(), 'Dataset hasn\'t been loaded'
    model = mades.ConditionalGaussianMade(data.n_labels, data.n_dims, n_hiddens, act_fun, mode=mode)
    train_cond(model, a_made)
    save_model(model, 'made_cond', mode, n_hiddens, act_fun, None, False)


def train_mog_made(n_hiddens, act_fun, n_comps, mode):

    assert is_data_loaded(), 'Dataset hasn\'t been loaded'
    model = mades.MixtureOfGaussiansMade(data.n_dims, n_hiddens, act_fun, n_comps, mode=mode)
    train(model, a_made)
    save_model(model, 'made', mode, n_hiddens, act_fun, n_comps, False)


def train_mog_made_cond(n_hiddens, act_fun, n_comps, mode):

    assert is_data_loaded(), 'Dataset hasn\'t been loaded'
    model = mades.ConditionalMixtureOfGaussiansMade(data.n_labels, data.n_dims, n_hiddens, act_fun, n_comps, mode=mode)
    train_cond(model, a_made)
    save_model(model, 'made_cond', mode, n_hiddens, act_fun, n_comps, False)


def train_realnvp(n_hiddens, s_act, t_act, n_layers):

    assert is_data_loaded(), 'Dataset hasn\'t been loaded'
    model = nvps.RealNVP(data.n_dims, n_hiddens, s_act, t_act, n_layers)
    train(model, a_flow)
    save_model(model, 'realnvp', None, n_hiddens, s_act + t_act, n_layers, True)


def train_realnvp_cond(n_hiddens, s_act, t_act, n_layers):

    assert is_data_loaded(), 'Dataset hasn\'t been loaded'
    model = nvps.ConditionalRealNVP(data.n_labels, data.n_dims, n_hiddens, s_act, t_act, n_layers)
    train_cond(model, a_flow)
    save_model(model, 'realnvp_cond', None, n_hiddens, s_act + t_act, n_layers, True)


def train_maf(n_hiddens, act_fun, n_mades, mode):

    assert is_data_loaded(), 'Dataset hasn\'t been loaded'
    model = mafs.MaskedAutoregressiveFlow(data.n_dims, n_hiddens, act_fun, n_mades, mode=mode)
    train(model, a_flow)
    save_model(model, 'maf', mode, n_hiddens, act_fun, n_mades, True)


def train_maf_cond(n_hiddens, act_fun, n_mades, mode):

    assert is_data_loaded(), 'Dataset hasn\'t been loaded'
    model = mafs.ConditionalMaskedAutoregressiveFlow(data.n_labels, data.n_dims, n_hiddens, act_fun, n_mades, mode=mode)
    train_cond(model, a_flow)
    save_model(model, 'maf_cond', mode, n_hiddens, act_fun, n_mades, True)


def train_maf_on_made(n_hiddens, act_fun, n_layers, n_comps, mode):

    assert is_data_loaded(), 'Dataset hasn\'t been loaded'
    model = mafs.MaskedAutoregressiveFlow_on_MADE(data.n_dims, n_hiddens, act_fun, n_layers, n_comps, mode=mode)
    train(model, a_flow)
    save_model(model, 'maf_on_made', mode, n_hiddens, act_fun, [n_layers, n_comps], True)


def train_maf_on_made_cond(n_hiddens, act_fun, n_layers, n_comps, mode):

    assert is_data_loaded(), 'Dataset hasn\'t been loaded'
    model = mafs.ConditionalMaskedAutoregressiveFlow_on_MADE(data.n_labels, data.n_dims, n_hiddens, act_fun, n_layers, n_comps, mode=mode)
    train_cond(model, a_flow)
    save_model(model, 'maf_on_made_cond', mode, n_hiddens, act_fun, [n_layers, n_comps], True)


def is_conditional(model):
    """
    Checks whether the given model is conditional or not.
    :param model: a model
    :return: boolean
    """

    if isinstance(model, mades.GaussianMade) \
            or isinstance(model, mades.MixtureOfGaussiansMade) \
            or isinstance(model, nvps.RealNVP) \
            or isinstance(model, mafs.MaskedAutoregressiveFlow) \
            or isinstance(model, mafs.MaskedAutoregressiveFlow_on_MADE):
        return False

    elif isinstance(model, mades.ConditionalGaussianMade) \
            or isinstance(model, mades.ConditionalMixtureOfGaussiansMade) \
            or isinstance(model, nvps.ConditionalRealNVP) \
            or isinstance(model, mafs.ConditionalMaskedAutoregressiveFlow) \
            or isinstance(model, mafs.ConditionalMaskedAutoregressiveFlow_on_MADE):
        return True

    else:
        raise TypeError('Wrong type of model.')


def evaluate(model, split, n_samples=None):
    """
    Evaluate a trained model.
    :param model: the model to evaluate. Can be any made, maf, or real nvp
    :param split: string, the data split to evaluate on. Must be 'trn', 'val' or 'tst'
    :param n_samples: number of samples to generate from the model, or None for no samples
    """

    assert is_data_loaded(), 'Dataset hasn\'t been loaded'

    # choose which data split to evaluate on
    data_split = getattr(data, split, None)
    if data_split is None:
        raise ValueError('Invalid data split')

    if is_conditional(model):

        # calculate log probability
        logprobs = model.eval([data_split.y, data_split.x])
        print('logprob(x|y) = {0:.2f} +/- {1:.2f}'.format(logprobs.mean(), 2 * logprobs.std() / np.sqrt(data_split.N)))

        # classify test set
        logprobs = np.empty([data_split.N, data.n_labels])
        for i in range(data.n_labels):
            y = np.zeros([data_split.N, data.n_labels])
            y[:, i] = 1
            logprobs[:, i] = model.eval([y, data_split.x])
        predict_label = np.argmax(logprobs, axis=1)
        accuracy = (predict_label == data_split.labels).astype(float)
        logprobs = scipy.misc.logsumexp(logprobs, axis=1) - np.log(logprobs.shape[1])
        print('logprob(x) = {0:.2f} +/- {1:.2f}'.format(logprobs.mean(), 2 * logprobs.std() / np.sqrt(data_split.N)))
        print('classification accuracy = {0:.2%} +/- {1:.2%}'.format(accuracy.mean(), 2 * accuracy.std() / np.sqrt(data_split.N)))

        # generate data conditioned on label
        if n_samples is not None:
            for i in range(data.n_labels):

                # generate samples and sort according to log prob
                y = np.zeros(data.n_labels)
                y[i] = 1
                samples = model.gen(y, n_samples)
                lp_samples = model.eval([np.tile(y, [n_samples, 1]), samples])
                lp_samples = lp_samples[np.logical_not(np.isnan(lp_samples))]
                idx = np.argsort(lp_samples)
                samples = samples[idx][::-1]

                if data_name == 'mnist':
                    samples = (util.logistic(samples) - data.alpha) / (1 - 2*data.alpha)

                elif data_name == 'bsds300':
                    samples = np.hstack([samples, -np.sum(samples, axis=1)[:, np.newaxis]])

                elif data_name == 'cifar10':
                    samples = (util.logistic(samples) - data.alpha) / (1 - 2*data.alpha)
                    D = int(data.n_dims / 3)
                    r = samples[:, :D]
                    g = samples[:, D:2*D]
                    b = samples[:, 2*D:]
                    samples = np.stack([r, g, b], axis=2)

                else:
                    raise ValueError('non-image dataset')

                util.disp_imdata(samples, data.image_size, [5, 8])

    else:

        # calculate average log probability
        logprobs = model.eval(data_split.x)
        print('logprob(x) = {0:.2f} +/- {1:.2f}'.format(logprobs.mean(), 2 * logprobs.std() / np.sqrt(data_split.N)))

        # generate data
        if n_samples is not None:

            # generate samples and sort according to log prob
            samples = model.gen(n_samples)
            lp_samples = model.eval(samples)
            lp_samples = lp_samples[np.logical_not(np.isnan(lp_samples))]
            idx = np.argsort(lp_samples)
            samples = samples[idx][::-1]

            if data_name == 'mnist':
                samples = (util.logistic(samples) - data.alpha) / (1 - 2*data.alpha)

            elif data_name == 'bsds300':
                samples = np.hstack([samples, -np.sum(samples, axis=1)[:, np.newaxis]])

            elif data_name == 'cifar10':
                samples = (util.logistic(samples) - data.alpha) / (1 - 2*data.alpha)
                D = int(data.n_dims / 3)
                r = samples[:, :D]
                g = samples[:, D:2*D]
                b = samples[:, 2*D:]
                samples = np.stack([r, g, b], axis=2)

            else:
                raise ValueError('non-image dataset')

            util.disp_imdata(samples, data.image_size, [5, 8])

    plt.show()


def evaluate_logprob(model, split, use_image_space=False, return_avg=True, batch=2000):
    """
    Evaluate a trained model only in terms of log probability.
    :param model: the model to evaluate. Can be any made, maf, or real nvp
    :param split: string, the data split to evaluate on. Must be 'trn', 'val' or 'tst'
    :param use_image_space: bool, whether to report log probability in [0, 1] image space (only for cifar and mnist)
    :param return_avg: bool, whether to return average log prob with std error, or all log probs
    :param batch: batch size to use for computing log probability
    :return: average log probability & standard error, or all log probs
    """

    assert is_data_loaded(), 'Dataset hasn\'t been loaded'

    # choose which data split to evaluate on
    data_split = getattr(data, split, None)
    if data_split is None:
        raise ValueError('Invalid data split')

    if is_conditional(model):

        logprobs = np.empty([data_split.N, data.n_labels])

        for i in range(data.n_labels):

            # create labels
            y = np.zeros([data_split.N, data.n_labels])
            y[:, i] = 1

            # process data in batches to make sure they fit in memory
            r, l = 0, batch
            while r < data_split.N:
                logprobs[r:l, i] = model.eval([y[r:l], data_split.x[r:l]])
                l += batch
                r += batch

        logprobs = scipy.misc.logsumexp(logprobs, axis=1) - np.log(logprobs.shape[1])

    else:

        logprobs = np.empty(data_split.N)

        # process data in batches to make sure they fit in memory
        r, l = 0, batch
        while r < data_split.N:
            logprobs[r:l] = model.eval(data_split.x[r:l])
            l += batch
            r += batch

    if use_image_space:
        assert data_name in ['mnist', 'cifar10']
        z = util.logistic(data_split.x)
        logprobs += data.n_dims * np.log(1-2*data.alpha) - np.sum(np.log(z) + np.log(1-z), axis=1)

    if return_avg:
        avg_logprob = logprobs.mean()
        std_err = logprobs.std() / np.sqrt(data_split.N)
        return avg_logprob, std_err

    else:
        return logprobs


def evaluate_random_numbers(model, split, n_marginals=5):
    """
    Evaluates the model by looking at the distribution of the random numbers for some data split. The more gaussian it
    look, the better the model fits the data.
    :param model: the model
    :param split: the data split, must be 'trn', 'val', or 'tst'
    :param n_marginals: number of marginal histograms of random numbers to plot
    """

    assert is_data_loaded(), 'Dataset hasn\'t been loaded'

    # choose which data split to use
    data_split = getattr(data, split, None)
    if data_split is None:
        raise ValueError('Invalid data split')

    # determine whether model is conditional
    if is_conditional(model):
        x = [data_split.y, data_split.x]
    else:
        x = data_split.x

    # calculate random numbers
    u = model.calc_random_numbers(x)

    # estimate kl to unit gaussian
    q = pdfs.fit_gaussian(u)
    p = pdfs.Gaussian(m=np.zeros(data.n_dims), S=np.eye(data.n_dims))
    print('KL(q||p) = {0:.2f}'.format(q.kl(p)))

    # plot some marginals
    util.plot_hist_marginals(u[:, :n_marginals])
    plt.show()


def fit_and_evaluate_gaussian(split, cond=False, use_image_space=False, return_avg=True):
    """
    Fits a gaussian to the train data and evaluates it on the given split.
    :param split: the data split to evaluate on, must be 'trn', 'val', or 'tst'
    :param cond: boolean, whether to fit a gaussian per conditional
    :param use_image_space: bool, whether to report log probability in [0, 1] image space (only for cifar and mnist)
    :param return_avg: bool, whether to return average log prob with std error, or all log probs
    :return: average log probability & standard error, or all lop probs
    """

    assert is_data_loaded(), 'Dataset hasn\'t been loaded'

    # choose which data split to evaluate on
    data_split = getattr(data, split, None)
    if data_split is None:
        raise ValueError('Invalid data split')

    if cond:
        comps = []
        for i in range(data.n_labels):
            idx = data.trn.labels == i
            comp = pdfs.fit_gaussian(data.trn.x[idx])
            comps.append(comp)
        prior = np.ones(data.n_labels, dtype=float) / data.n_labels
        model = pdfs.MoG(prior, xs=comps)

    else:
        model = pdfs.fit_gaussian(data.trn.x)

    logprobs = model.eval(data_split.x)

    if use_image_space:
        assert data_name in ['mnist', 'cifar10']
        z = util.logistic(data_split.x)
        logprobs += data.n_dims * np.log(1-2*data.alpha) - np.sum(np.log(z) + np.log(1-z), axis=1)

    if return_avg:
        avg_logprob = logprobs.mean()
        std_err = logprobs.std() / np.sqrt(data_split.N)
        return avg_logprob, std_err

    else:
        return logprobs
