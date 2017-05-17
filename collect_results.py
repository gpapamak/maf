# collects all saved results from experiments on given datasets
#
# USAGE:
#   python collect_results.py dataset1 dataset2 ...


import sys
import itertools
import numpy as np
import experiments as ex
import util


split = 'tst'              # choose which data split to evaluate on 'trn', 'val' or 'tst'
n_err = 2                  # number of stds in error bars
bits_per_pixel = False     # whether to use bits/pixel instead of log likelihood (only for image datasets)

root_results = 'results/'  # folder where to save results


def calc_bits_per_pixel(lp, lp_err):

    Dlog2 = ex.data.n_dims * np.log(2)
    bpp = -lp / Dlog2 + 8
    bpp_err = lp_err / Dlog2

    return bpp, bpp_err


def result(model_name, mode, n_hiddens, act_fun, n_comps=None, batch_norm=False):

    try:
        model = ex.load_model(model_name, mode, n_hiddens, act_fun, n_comps, batch_norm)

    except IOError:
        return 'N/A'

    res, err = ex.evaluate_logprob(model, split, use_image_space=bits_per_pixel)

    if bits_per_pixel:
        res, err = calc_bits_per_pixel(res, err)

    return '{0:.2f} +/- {1:.2f}'.format(res, n_err * err)


def collect_results(data, n_hiddens, n_layers, n_comps, n_layers_comps, act_funs, modes, has_cond):

    print 'collecting for {0}...'.format(data)
    ex.load_data(data)

    # create file to write to
    filename = ('{0}_{1}_bpp.txt' if bits_per_pixel else '{0}_{1}.txt').format(data, split)
    util.make_folder(root_results)
    f = open(root_results + filename, 'w')
    f.write('Results for {0}\n'.format(data))
    f.write('\n')

    for act, mode in itertools.product(act_funs, modes):

        f.write('actf: {0}\n'.format(act))
        f.write('mode: {0}\n'.format(mode))
        f.write('\n')

        # gaussian
        f.write('Gaussian\n')
        res, err = ex.fit_and_evaluate_gaussian(split, cond=False, use_image_space=bits_per_pixel)
        if bits_per_pixel:
            res, err = calc_bits_per_pixel(res, err)
        f.write('  {0:.2f} +/- {1:.2f}\n'.format(res, n_err * err))
        if has_cond:
            f.write('conditional\n')
            res, err = ex.fit_and_evaluate_gaussian(split, cond=True, use_image_space=bits_per_pixel)
            if bits_per_pixel:
                res, err = calc_bits_per_pixel(res, err)
            f.write('  {0:.2f} +/- {1:.2f}\n'.format(res, n_err * err))
        f.write('\n')

        # made
        f.write('MADE 1 comp\n')
        for nh in n_hiddens:
            f.write('  [1 x {0}]: {1}\n'.format(nh, result('made', mode, [nh]*1, act)))
            f.write('  [2 x {0}]: {1}\n'.format(nh, result('made', mode, [nh]*2, act)))
        if has_cond:
            f.write('conditional\n')
            for nh in n_hiddens:
                f.write('  [1 x {0}]: {1}\n'.format(nh, result('made_cond', mode, [nh]*1, act)))
                f.write('  [2 x {0}]: {1}\n'.format(nh, result('made_cond', mode, [nh]*2, act)))
        f.write('\n')

        # mog made
        for nc in n_comps:
            f.write('MADE {0} comp\n'.format(nc))
            for nh in n_hiddens:
                f.write('  [1 x {0}]: {1}\n'.format(nh, result('made', mode, [nh]*1, act, nc)))
                f.write('  [2 x {0}]: {1}\n'.format(nh, result('made', mode, [nh]*2, act, nc)))
            if has_cond:
                f.write('conditional\n')
                for nh in n_hiddens:
                    f.write('  [1 x {0}]: {1}\n'.format(nh, result('made_cond', mode, [nh]*1, act, nc)))
                    f.write('  [2 x {0}]: {1}\n'.format(nh, result('made_cond', mode, [nh]*2, act, nc)))
            f.write('\n')

        # real nvp
        for nl in n_layers:
            f.write('RealNVP {0} layers\n'.format(nl))
            for nh in n_hiddens:
                f.write('  [1 x {0}]: {1}\n'.format(nh, result('realnvp', None, [nh]*1, 'tanhrelu', nl, True)))
                f.write('  [2 x {0}]: {1}\n'.format(nh, result('realnvp', None, [nh]*2, 'tanhrelu', nl, True)))
            if has_cond:
                f.write('conditional\n')
                for nh in n_hiddens:
                    f.write('  [1 x {0}]: {1}\n'.format(nh, result('realnvp_cond', None, [nh]*1, 'tanhrelu', nl, True)))
                    f.write('  [2 x {0}]: {1}\n'.format(nh, result('realnvp_cond', None, [nh]*2, 'tanhrelu', nl, True)))
            f.write('\n')

        # maf
        for nl in n_layers:
            f.write('MAF {0} layers\n'.format(nl))
            for nh in n_hiddens:
                f.write('  [1 x {0}]: {1}\n'.format(nh, result('maf', mode, [nh]*1, act, nl, True)))
                f.write('  [2 x {0}]: {1}\n'.format(nh, result('maf', mode, [nh]*2, act, nl, True)))
            if has_cond:
                f.write('conditional\n')
                for nh in n_hiddens:
                    f.write('  [1 x {0}]: {1}\n'.format(nh, result('maf_cond', mode, [nh]*1, act, nl, True)))
                    f.write('  [2 x {0}]: {1}\n'.format(nh, result('maf_cond', mode, [nh]*2, act, nl, True)))
            f.write('\n')

        # maf on made
        for nl, nc in n_layers_comps:
            f.write('MAF {0} layers on MADE {1} comp\n'.format(nl, nc))
            for nh in n_hiddens:
                f.write('  [1 x {0}]: {1}\n'.format(nh, result('maf_on_made', mode, [nh]*1, act, [nl, nc], True)))
                f.write('  [2 x {0}]: {1}\n'.format(nh, result('maf_on_made', mode, [nh]*2, act, [nl, nc], True)))
            if has_cond:
                f.write('conditional\n')
                for nh in n_hiddens:
                    f.write('  [1 x {0}]: {1}\n'.format(nh, result('maf_on_made_cond', mode, [nh]*1, act, [nl, nc], True)))
                    f.write('  [2 x {0}]: {1}\n'.format(nh, result('maf_on_made_cond', mode, [nh]*2, act, [nl, nc], True)))
            f.write('\n')

    # close file
    f.close()


def main():

    for data in sys.argv[1:]:

        if data == 'power':
            collect_results(data, [100], [5, 10], [10], [(5, 10)], ['relu'], ['sequential'], False)

        elif data == 'gas':
            collect_results(data, [100], [5, 10], [10], [(5, 10)], ['tanh'], ['sequential'], False)

        elif data == 'hepmass':
            collect_results(data, [512], [5, 10], [10], [(5, 10)], ['relu'], ['sequential'], False)

        elif data == 'miniboone':
            collect_results(data, [512], [5, 10], [10], [(5, 10)], ['relu'], ['sequential'], False)

        elif data == 'bsds300':
            collect_results(data, [512, 1024], [5, 10], [10], [(5, 10)], ['relu'], ['sequential'], False)

        elif data == 'mnist':
            collect_results(data, [1024], [5, 10], [10], [(5, 10)], ['relu'], ['sequential'], True)

        elif data == 'cifar10':
            collect_results(data, [1024, 2048], [5, 10], [10], [(5, 10)], ['relu'], ['random'], True)

        else:
            print '{0} is not a valid dataset'.format(data)
            continue


if __name__ == '__main__':
    main()
