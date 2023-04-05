import numpy as np
import theano
import theano.tensor as tt
import matplotlib.pyplot as plt

import ml.step_strategies as ss
import ml.data_streams as ds
import util


dtype = theano.config.floatX


class SGD:

    def __init__(self, model, trn_data, trn_loss, trn_target=None, val_data=None, val_loss=None, val_target=None, step=ss.Adam()):
        """
        Constructs and configures the trainer.
        :param model: the model to be trained
        :param trn_data: train inputs and (possibly) train targets
        :param trn_loss: theano variable representing the train loss to minimize
        :param trn_target: theano variable representing the train target
        :param val_data: validation inputs and (possibly) validation targets
        :param val_loss: theano variable representing the validation loss
        :param val_target: theano variable representing the validation target
        :param step: step size strategy object
        :return: None
        """

        # parse input
        # TODO: it would be good to type check the other inputs too
        assert isinstance(step, ss.StepStrategy), 'Step must be a step strategy object.'

        # prepare train data
        n_trn_data_list = set([x.shape[0] for x in trn_data])
        assert len(n_trn_data_list) == 1, 'Number of train data is not consistent.'
        self.n_trn_data = list(n_trn_data_list)[0]
        trn_data = [theano.shared(x.astype(dtype), borrow=True) for x in trn_data]

        # compile theano function for a single training update
        grads = tt.grad(trn_loss, model.parms)
        idx = tt.ivector('idx')
        trn_inputs = [model.input] if trn_target is None else [model.input, trn_target]
        self.make_update = theano.function(
            inputs=[idx],
            outputs=trn_loss,
            givens=zip(trn_inputs, [x[idx] for x in trn_data]),
            updates=step.updates(model.parms, grads)
        )

        # if model uses batch norm, compile a theano function for setting up stats
        if getattr(model, 'batch_norm', False):
            batch_norm_givens = [(bn.m, bn.bm) for bn in model.bns] + [(bn.v, bn.bv) for bn in model.bns]
            self.set_batch_norm_stats = theano.function(
                inputs=[],
                givens=zip(trn_inputs, trn_data),
                updates=[(bn.bm, bn.m) for bn in model.bns] + [(bn.bv, bn.v) for bn in model.bns]
            )
        else:
            self.set_batch_norm_stats = None
            batch_norm_givens = []

        # if validation data is given, then set up validation too
        self.do_validation = val_data is not None

        if self.do_validation:

            # prepare validation data
            n_val_data_list = set([x.shape[0] for x in val_data])
            assert len(n_val_data_list) == 1, 'Number of validation data is not consistent.'
            self.n_val_data = list(n_val_data_list)[0]
            val_data = [theano.shared(x.astype(dtype), borrow=True) for x in val_data]

            # compile theano function for validation
            val_inputs = [model.input] if val_target is None else [model.input, val_target]
            self.validate = theano.function(
                inputs=[],
                outputs=val_loss,
                givens=zip(val_inputs, val_data) + batch_norm_givens
            )

            # create checkpointer to store best model
            self.checkpointer = ModelCheckpointer(model)
            self.best_val_loss = float('inf')

        # initialize some variables
        self.trn_loss = float('inf')
        self.idx_stream = ds.IndexSubSampler(self.n_trn_data)

    def train(self, minibatch=None, tol=None, maxepochs=None, monitor_every=None, patience=None, verbose=True, show_progress=False, val_in_same_plot=True):
        """
        Trains the model.
        :param minibatch: minibatch size
        :param tol: tolerance
        :param maxepochs: maximum number of epochs
        :param monitor_every: monitoring frequency
        :param patience: maximum number of validation steps to wait for improvement before early stopping
        :param verbose: if True, print progress during training
        :param show_progress: if True, plot training and validation progress
        :param val_in_same_plot: if True, plot validation progress in same plot as training progress
        :return: None
        """

        # parse input
        assert minibatch is None or util.isposint(minibatch), 'Minibatch size must be a positive integer or None.'
        assert tol is None or tol > 0.0, 'Tolerance must be positive or None.'
        assert maxepochs is None or maxepochs > 0.0, 'Maximum number of epochs must be positive or None.'
        assert monitor_every is None or monitor_every > 0.0, 'Monitoring frequency must be positive or None.'
        assert patience is None or util.isposint(patience), 'Patience must be a positive integer or None.'
        assert isinstance(verbose, bool), 'verbose must be boolean.'
        assert isinstance(show_progress, bool), 'store_progress must be boolean.'
        assert isinstance(val_in_same_plot, bool), 'val_in_same_plot must be boolean.'

        # initialize some variables
        iter = 0
        progress_epc = []
        progress_trn = []
        progress_val = []
        minibatch = self.n_trn_data if minibatch is None else minibatch
        maxiter = float('inf') if maxepochs is None else np.ceil(maxepochs * self.n_trn_data / float(minibatch))
        monitor_every = float('inf') if monitor_every is None else np.ceil(monitor_every * self.n_trn_data / float(minibatch))
        patience = float('inf') if patience is None else patience
        patience_left = patience
        best_epoch = None

        # main training loop
        while True:

            # make update to parameters
            trn_loss = self.make_update(self.idx_stream.gen(minibatch))
            diff = self.trn_loss - trn_loss
            iter += 1
            self.trn_loss = trn_loss

            if iter % monitor_every == 0:

                epoch = iter * float(minibatch) / self.n_trn_data

                # do validation
                if self.do_validation:
                    if self.set_batch_norm_stats is not None: self.set_batch_norm_stats()
                    val_loss = self.validate()
                    patience_left -= 1

                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.checkpointer.checkpoint()
                        best_epoch = epoch
                        patience_left = patience

                # monitor progress
                if show_progress:
                    progress_epc.append(epoch)
                    progress_trn.append(trn_loss)
                    if self.do_validation: progress_val.append(val_loss)

                # print info
                if verbose:
                    if self.do_validation:
                        print('Epoch = {0:.2f}, train loss = {1}, validation loss = {2}'.format(epoch, trn_loss, val_loss))
                    else:
                        print('Epoch = {0:.2f}, train loss = {1}'.format(epoch, trn_loss))

            # check for convergence
            if abs(diff) < tol or iter >= maxiter or patience_left <= 0:
                if self.do_validation: self.checkpointer.restore()
                if self.set_batch_norm_stats is not None: self.set_batch_norm_stats()
                break

        # plot progress
        if show_progress:

            if self.do_validation:

                if val_in_same_plot:
                    fig, ax = plt.subplots(1, 1)
                    ax.semilogx(progress_epc, progress_trn, 'b', label='training')
                    ax.semilogx(progress_epc, progress_val, 'r', label='validation')
                    ax.vlines(best_epoch, ax.get_ylim()[0], ax.get_ylim()[1], color='g', linestyles='dashed', label='best')
                    ax.set_xlabel('epochs')
                    ax.set_ylabel('loss')
                    ax.legend()

                else:
                    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
                    ax1.semilogx(progress_epc, progress_trn, 'b')
                    ax2.semilogx(progress_epc, progress_val, 'r')
                    ax1.vlines(best_epoch, ax1.get_ylim()[0], ax1.get_ylim()[1], color='g', linestyles='dashed', label='best')
                    ax2.vlines(best_epoch, ax2.get_ylim()[0], ax2.get_ylim()[1], color='g', linestyles='dashed', label='best')
                    ax2.set_xlabel('epochs')
                    ax1.set_ylabel('training loss')
                    ax2.set_ylabel('validation loss')

            else:
                fig, ax = plt.subplots(1, 1)
                ax.semilogx(progress_epc, progress_trn, 'b')
                ax.set_xlabel('epochs')
                ax.set_ylabel('training loss')
                ax.legend()

            plt.show(block=False)


class ModelCheckpointer:
    """
    Helper class which makes checkpoints of a given model.
    Currently one checkpoint is supported; checkpointing twice overwrites previous checkpoint.
    """

    def __init__(self, model):
        """
        :param model: A machine learning model to be checkpointed.
        """
        self.model = model
        self.checkpointed_parms = [np.empty_like(p.get_value()) for p in model.parms]

    def checkpoint(self):
        """
        Checkpoints current model. Overwrites previous checkpoint.
        """
        for i, p in enumerate(self.model.parms):
            self.checkpointed_parms[i] = p.get_value().copy()

    def restore(self):
        """
        Restores last checkpointed model.
        """
        for i, p in enumerate(self.checkpointed_parms):
            self.model.parms[i].set_value(p)
