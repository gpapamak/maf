# runs all experiments and saves all results for given datasets
#
# USAGE:
#   python run_experiments.py dataset1 dataset2 ...


import sys
import experiments as ex


def run_experiments_power():

    n_hiddens = 100
    n_layers = 5
    n_comps = 10
    act_fun = 'relu'
    mode = 'sequential'

    ex.load_data('power')

    ex.train_made([n_hiddens]*1, act_fun, mode)
    ex.train_made([n_hiddens]*2, act_fun, mode)

    ex.train_mog_made([n_hiddens]*1, act_fun, n_comps, mode)
    ex.train_mog_made([n_hiddens]*2, act_fun, n_comps, mode)

    ex.train_realnvp([n_hiddens]*1, 'tanh', 'relu', n_layers*1)
    ex.train_realnvp([n_hiddens]*2, 'tanh', 'relu', n_layers*1)
    ex.train_realnvp([n_hiddens]*1, 'tanh', 'relu', n_layers*2)
    ex.train_realnvp([n_hiddens]*2, 'tanh', 'relu', n_layers*2)

    ex.train_maf([n_hiddens]*1, act_fun, n_layers*1, mode)
    ex.train_maf([n_hiddens]*2, act_fun, n_layers*1, mode)
    ex.train_maf([n_hiddens]*1, act_fun, n_layers*2, mode)
    ex.train_maf([n_hiddens]*2, act_fun, n_layers*2, mode)

    ex.train_maf_on_made([n_hiddens]*1, act_fun, n_layers, n_comps, mode)
    ex.train_maf_on_made([n_hiddens]*2, act_fun, n_layers, n_comps, mode)


def run_experiments_gas():

    n_hiddens = 100
    n_layers = 5
    n_comps = 10
    act_fun = 'tanh'
    mode = 'sequential'

    ex.load_data('gas')

    ex.train_made([n_hiddens]*1, act_fun, mode)
    ex.train_made([n_hiddens]*2, act_fun, mode)

    ex.train_mog_made([n_hiddens]*1, act_fun, n_comps, mode)
    ex.train_mog_made([n_hiddens]*2, act_fun, n_comps, mode)

    ex.train_realnvp([n_hiddens]*1, 'tanh', 'relu', n_layers*1)
    ex.train_realnvp([n_hiddens]*2, 'tanh', 'relu', n_layers*1)
    ex.train_realnvp([n_hiddens]*1, 'tanh', 'relu', n_layers*2)
    ex.train_realnvp([n_hiddens]*2, 'tanh', 'relu', n_layers*2)

    ex.train_maf([n_hiddens]*1, act_fun, n_layers*1, mode)
    ex.train_maf([n_hiddens]*2, act_fun, n_layers*1, mode)
    ex.train_maf([n_hiddens]*1, act_fun, n_layers*2, mode)
    ex.train_maf([n_hiddens]*2, act_fun, n_layers*2, mode)

    ex.train_maf_on_made([n_hiddens]*1, act_fun, n_layers, n_comps, mode)
    ex.train_maf_on_made([n_hiddens]*2, act_fun, n_layers, n_comps, mode)


def run_experiments_hepmass():

    n_hiddens = 512
    n_layers = 5
    n_comps = 10
    act_fun = 'relu'
    mode = 'sequential'

    ex.load_data('hepmass')

    ex.train_made([n_hiddens]*1, act_fun, mode)
    ex.train_made([n_hiddens]*2, act_fun, mode)

    ex.train_mog_made([n_hiddens]*1, act_fun, n_comps, mode)
    ex.train_mog_made([n_hiddens]*2, act_fun, n_comps, mode)

    ex.train_realnvp([n_hiddens]*1, 'tanh', 'relu', n_layers*1)
    ex.train_realnvp([n_hiddens]*2, 'tanh', 'relu', n_layers*1)
    ex.train_realnvp([n_hiddens]*1, 'tanh', 'relu', n_layers*2)
    ex.train_realnvp([n_hiddens]*2, 'tanh', 'relu', n_layers*2)

    ex.train_maf([n_hiddens]*1, act_fun, n_layers*1, mode)
    ex.train_maf([n_hiddens]*2, act_fun, n_layers*1, mode)
    ex.train_maf([n_hiddens]*1, act_fun, n_layers*2, mode)
    ex.train_maf([n_hiddens]*2, act_fun, n_layers*2, mode)

    ex.train_maf_on_made([n_hiddens]*1, act_fun, n_layers, n_comps, mode)
    ex.train_maf_on_made([n_hiddens]*2, act_fun, n_layers, n_comps, mode)


def run_experiments_miniboone():

    n_hiddens = 512
    n_layers = 5
    n_comps = 10
    act_fun = 'relu'
    mode = 'sequential'

    ex.load_data('miniboone')

    ex.train_made([n_hiddens]*1, act_fun, mode)
    ex.train_made([n_hiddens]*2, act_fun, mode)

    ex.train_mog_made([n_hiddens]*1, act_fun, n_comps, mode)
    ex.train_mog_made([n_hiddens]*2, act_fun, n_comps, mode)

    ex.train_realnvp([n_hiddens]*1, 'tanh', 'relu', n_layers*1)
    ex.train_realnvp([n_hiddens]*2, 'tanh', 'relu', n_layers*1)
    ex.train_realnvp([n_hiddens]*1, 'tanh', 'relu', n_layers*2)
    ex.train_realnvp([n_hiddens]*2, 'tanh', 'relu', n_layers*2)

    ex.train_maf([n_hiddens]*1, act_fun, n_layers*1, mode)
    ex.train_maf([n_hiddens]*2, act_fun, n_layers*1, mode)
    ex.train_maf([n_hiddens]*1, act_fun, n_layers*2, mode)
    ex.train_maf([n_hiddens]*2, act_fun, n_layers*2, mode)

    ex.train_maf_on_made([n_hiddens]*1, act_fun, n_layers, n_comps, mode)
    ex.train_maf_on_made([n_hiddens]*2, act_fun, n_layers, n_comps, mode)


def run_experiments_bsds300():

    n_layers = 5
    n_comps = 10
    act_fun = 'relu'
    mode = 'sequential'

    ex.load_data('bsds300')

    for n_hiddens in [512, 1024]:

        ex.train_made([n_hiddens]*1, act_fun, mode)
        ex.train_made([n_hiddens]*2, act_fun, mode)

        ex.train_mog_made([n_hiddens]*1, act_fun, n_comps, mode)
        ex.train_mog_made([n_hiddens]*2, act_fun, n_comps, mode)

        ex.train_realnvp([n_hiddens]*1, 'tanh', 'relu', n_layers*1)
        ex.train_realnvp([n_hiddens]*2, 'tanh', 'relu', n_layers*1)
        ex.train_realnvp([n_hiddens]*1, 'tanh', 'relu', n_layers*2)
        ex.train_realnvp([n_hiddens]*2, 'tanh', 'relu', n_layers*2)

        ex.train_maf([n_hiddens]*1, act_fun, n_layers*1, mode)
        ex.train_maf([n_hiddens]*2, act_fun, n_layers*1, mode)
        ex.train_maf([n_hiddens]*1, act_fun, n_layers*2, mode)
        ex.train_maf([n_hiddens]*2, act_fun, n_layers*2, mode)

        ex.train_maf_on_made([n_hiddens]*1, act_fun, n_layers, n_comps, mode)
        ex.train_maf_on_made([n_hiddens]*2, act_fun, n_layers, n_comps, mode)


def run_experiments_mnist():

    n_hiddens = 1024
    n_layers = 5
    n_comps = 10
    act_fun = 'relu'
    mode = 'sequential'

    ex.load_data('mnist')

    ex.train_made([n_hiddens]*2, act_fun, mode)
    ex.train_made_cond([n_hiddens]*2, act_fun, mode)

    ex.train_mog_made([n_hiddens]*2, act_fun, n_comps, mode)
    ex.train_mog_made_cond([n_hiddens]*2, act_fun, n_comps, mode)

    for i in [1, 2]:

        ex.train_realnvp([n_hiddens]*2, 'tanh', 'relu', n_layers*i)
        ex.train_realnvp_cond([n_hiddens]*2, 'tanh', 'relu', n_layers*i)

        ex.train_maf([n_hiddens]*2, act_fun, n_layers*i, mode)
        ex.train_maf_cond([n_hiddens]*2, act_fun, n_layers*i, mode)

    ex.train_maf_on_made([n_hiddens]*2, act_fun, n_layers, n_comps, mode)
    ex.train_maf_on_made_cond([n_hiddens]*2, act_fun, n_layers, n_comps, mode)


def run_experiments_cifar10():

    n_layers = 5
    n_comps = 10
    act_fun = 'relu'
    mode = 'random'

    ex.load_data('cifar10')

    for n_hiddens in [1024, 2048]:

        ex.train_made([n_hiddens]*1, act_fun, mode)
        ex.train_made([n_hiddens]*2, act_fun, mode)
        ex.train_made_cond([n_hiddens]*1, act_fun, mode)
        ex.train_made_cond([n_hiddens]*2, act_fun, mode)

        ex.train_mog_made([n_hiddens]*1, act_fun, n_comps, mode)
        ex.train_mog_made([n_hiddens]*2, act_fun, n_comps, mode)
        ex.train_mog_made_cond([n_hiddens]*1, act_fun, n_comps, mode)
        ex.train_mog_made_cond([n_hiddens]*2, act_fun, n_comps, mode)

        for i in [1, 2]:

            ex.train_realnvp([n_hiddens]*1, 'tanh', 'relu', n_layers*i)
            ex.train_realnvp([n_hiddens]*2, 'tanh', 'relu', n_layers*i)
            ex.train_realnvp_cond([n_hiddens]*1, 'tanh', 'relu', n_layers*i)
            ex.train_realnvp_cond([n_hiddens]*2, 'tanh', 'relu', n_layers*i)

            ex.train_maf([n_hiddens]*1, act_fun, n_layers*i, mode)
            ex.train_maf([n_hiddens]*2, act_fun, n_layers*i, mode)
            ex.train_maf_cond([n_hiddens]*1, act_fun, n_layers*i, mode)
            ex.train_maf_cond([n_hiddens]*2, act_fun, n_layers*i, mode)

        ex.train_maf_on_made([n_hiddens]*1, act_fun, n_layers, n_comps, mode)
        ex.train_maf_on_made([n_hiddens]*2, act_fun, n_layers, n_comps, mode)
        ex.train_maf_on_made_cond([n_hiddens]*1, act_fun, n_layers, n_comps, mode)
        ex.train_maf_on_made_cond([n_hiddens]*2, act_fun, n_layers, n_comps, mode)


def main():

    methods = dict()
    methods['power'] = run_experiments_power
    methods['gas'] = run_experiments_gas
    methods['hepmass'] = run_experiments_hepmass
    methods['miniboone'] = run_experiments_miniboone
    methods['bsds300'] = run_experiments_bsds300
    methods['mnist'] = run_experiments_mnist
    methods['cifar10'] = run_experiments_cifar10

    for name in sys.argv[1:]:
        methods[name]()


if __name__ == '__main__':
    main()
