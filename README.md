# Masked Autoregressive Flow for Density Estimation

Code for reproducing the experiments in the paper:

> G. Papamakarios, T. Pavlakou, I. Murray, _Masked Autoregressive Flow for Density Estimation_, NeurIPS 2017.</br>
> [[arXiv]](https://arxiv.org/abs/1705.07057) [[bibtex]](https://gpapamak.github.io/bibtex/maf.bib)

## How to run the code

To run all experiments for a particular dataset, run:

```
python run_experiments.py <dataset>
```

This will train and save all models associated with that dataset.

To evaluate all trained models and collect the results in a text file, run:

```
python collect_results.py <dataset>
```

In the above commands, `<dataset>` can be any of the following:
* `power`
* `gas`
* `hepmass`
* `miniboone`
* `bsds300`
* `mnist`
* `cifar10`

You can use the commands with more than one datasets as arguments separated by a space, for example:

```
python run_experiments.py mnist cifar10  
python collect_results.py mnist cifar10
```

## How to get the datasets

1. Downdload the datasets from: https://zenodo.org/record/1161203#.Wmtf_XVl8eN
2. Unpack the downloaded file, and place it in the same folder as the code.
3. Make sure the code reads from the location the datasets are saved at.
4. Run the code as described above.

All datasets used in the experiments are preprocessed versions of public datasets. None of them belongs to us. The original datasets are:

* POWER:  
  http://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption
  
* GAS  
  http://archive.ics.uci.edu/ml/datasets/Gas+sensor+array+under+dynamic+gas+mixtures
  
* HEPMASS  
  http://archive.ics.uci.edu/ml/datasets/HEPMASS
  
* MINIBOONE  
  http://archive.ics.uci.edu/ml/datasets/MiniBooNE+particle+identification
  
* BSDS300  
  https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/
  
* MNIST  
  http://yann.lecun.com/exdb/mnist/
  
* CIFAR-10  
  https://www.cs.toronto.edu/~kriz/cifar.html

