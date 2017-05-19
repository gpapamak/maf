# Masked Autoregressive Flow for Density Estimation

Code for reproducing the experiments in the paper:

> G. Papamakarios, T. Pavlakou, and I. Murray. _Masked Autoregressive Flow for Density Estimation_. 2017.

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

The datasets have to be separately downloaded from their public repositories, preprocessed as described in the paper, and placed in the folder the code reads from. The links to the public repositories for each dataset are:

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
  [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)

