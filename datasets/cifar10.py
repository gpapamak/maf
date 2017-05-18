from itertools import izip
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt

import datasets
import util


class CIFAR10:
    """
    The CIFAR-10 dataset.
    https://www.cs.toronto.edu/~kriz/cifar.html
    """

    alpha = 0.05

    class Data:
        """
        Constructs the dataset.
        """

        def __init__(self, x, l, logit, flip, dequantize, rng):

            D = x.shape[1] / 3                                 # number of pixels
            x = self._dequantize(x, rng) if dequantize else x  # dequantize
            x = self._logit_transform(x) if logit else x       # logit
            x = self._flip_augmentation(x) if flip else x      # flip
            self.x = x                                         # pixel values
            self.r = self.x[:, :D]                             # red component
            self.g = self.x[:, D:2*D]                          # green component
            self.b = self.x[:, 2*D:]                           # blue component
            self.labels = np.hstack([l, l]) if flip else l     # numeric labels
            self.y = util.one_hot_encode(self.labels, 10)      # 1-hot encoded labels
            self.N = self.x.shape[0]                           # number of datapoints

        @staticmethod
        def _dequantize(x, rng):
            """
            Adds noise to pixels to dequantize them.
            """
            return (x + rng.rand(*x.shape).astype(np.float32)) / 256.0

        @staticmethod
        def _logit_transform(x):
            """
            Transforms pixel values with logit to be unconstrained.
            """
            return util.logit(CIFAR10.alpha + (1 - 2*CIFAR10.alpha) * x)

        @staticmethod
        def _flip_augmentation(x):
            """
            Augments dataset x with horizontal flips.
            """
            D = x.shape[1] / 3
            I = int(np.sqrt(D))
            r = x[:,    :D].reshape([-1, I, I])[:, :, ::-1].reshape([-1, D])
            g = x[:, D:2*D].reshape([-1, I, I])[:, :, ::-1].reshape([-1, D])
            b = x[:,  2*D:].reshape([-1, I, I])[:, :, ::-1].reshape([-1, D])
            x_flip = np.hstack([r, g, b])
            return np.vstack([x, x_flip])

    def __init__(self, logit=False, flip=False, dequantize=True):

        rng = np.random.RandomState(42)

        path = datasets.root + 'cifar10/'

        # load train batches
        x = []
        l = []
        for i in xrange(1, 6):
            f = open(path + 'data_batch_' + str(i), 'rb')
            dict = pickle.load(f)
            x.append(dict['data'])
            l.append(dict['labels'])
            f.close()
        x = np.concatenate(x, axis=0)
        l = np.concatenate(l, axis=0)

        # use part of the train batches for validation
        split = int(0.9 * x.shape[0])
        self.trn = self.Data(x[:split], l[:split], logit, flip, dequantize, rng)
        self.val = self.Data(x[split:], l[split:], logit, False, dequantize, rng)

        # load test batch
        f = open(path + 'test_batch', 'rb')
        dict = pickle.load(f)
        x = dict['data']
        l = np.array(dict['labels'])
        f.close()
        self.tst = self.Data(x, l, logit, False, dequantize, rng)

        self.n_dims = self.trn.x.shape[1]
        self.n_labels = self.trn.y.shape[1]
        self.image_size = [int(np.sqrt(self.n_dims / 3))] * 2 + [3]

    def show_pixel_histograms(self, split, pixel=None):
        """
        Shows the histogram of pixel values, or of a specific pixel if given.
        """

        # get split
        data = getattr(self, split, None)
        if data is None:
            raise ValueError('Invalid data split')

        if pixel is None:
            data_r = data.r.flatten()
            data_g = data.g.flatten()
            data_b = data.b.flatten()

        else:
            row, col = pixel
            idx = row * self.image_size[0] + col
            data_r = data.r[:, idx]
            data_g = data.g[:, idx]
            data_b = data.b[:, idx]

        n_bins = int(np.sqrt(data.N))
        fig, axs = plt.subplots(3, 1)
        for ax, d, t in izip(axs, [data_r, data_g, data_b], ['red', 'green', 'blue']):
            ax.hist(d, n_bins, normed=True)
            ax.set_title(t)
        plt.show()

    def show_images(self, split):
        """
        Displays the images in a given split.
        :param split: string
        """

        # get split
        data = getattr(self, split, None)
        if data is None:
            raise ValueError('Invalid data split')

        # display images
        x = np.stack([data.r, data.g, data.b], axis=2)
        util.disp_imdata(x, self.image_size, [6, 10])

        plt.show()
