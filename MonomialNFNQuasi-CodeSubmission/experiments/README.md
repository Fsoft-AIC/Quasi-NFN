Follow the experiment instructions from [Permutation Equivariant Neural Functionals paper](https://github.com/AllanYangZhou/nfn)

The experiments require additional dependencies. In your virtual env:
```bash
pip install -r requirements.txt
pip install -e experiments/perceiver-pytorch
```
We use Python 3.8.16.

## Predicting generalization for small CNNs
### Getting CNN Zoo data
Download the [CIFAR10](https://storage.cloud.google.com/gresearch/smallcnnzoo-dataset/cifar10.tar.xz) data  (originally from [Unterthiner et al, 2020](https://github.com/google-research/google-research/tree/master/dnn_predict_accuracy)) into `./experiments/data`, and extract them. Change `data_root_dir` in `main.yaml` if you want to store the data somewhere else.

Download the [Augment data for CIFAR10](https://drive.google.com/file/d/1lpaRwbSunFKz_x-6HAa98ZqcSrG157ks/view?usp=sharing) into `./experiments` and extract them

### Monomial-NFN, no augment:
```
python -m experiments.launch_predict_gen seed=42 dset=zoo_cifar_aug nfnet=hnps_inv nfnet.head_cls.h_size=200 dset.filter_activation="relu" dset.augment_factor=1 
```

### Monomial-NFN Quasi, no augment:
```
python -m experiments.launch_predict_gen seed=42 dset=zoo_cifar_aug nfnet=hnps_inv_quasi nfnet.head_cls.h_size=200 dset.filter_activation="relu" dset.augment_factor=1 
```

### Monomial-NFN Quasi, with augment [1,10000]
```
python -m experiments.launch_predict_gen seed=42 dset=zoo_cifar_aug nfnet=hnps_inv_quasi nfnet.head_cls.h_size=200 dset.filter_activation="relu" dset.augment_factor=2 dset.scale_low=1 dset.scale_high=10000 dset.load_augmented=True dset.augment_path="experiments/augmented_data/augment_zoo_cifar_10000"
```

### For other baseline, no augment:
```
CUDA_VISIBLE_DEVICES=2 python -m experiments.launch_predict_gen seed=42 dset=zoo_cifar_aug nfnet=hnp_inv_sinkhorn  nfnet.head_cls.h_size=1000 dset.filter_activation="relu" dset.augment_factor=1 
```

Choices for nfnet:
* `hnp_inv` : HNP
* `np_inv` : NP
* `stat` : STATNN
* `hnps_inv` : Monomial-NFN
* `hnps_inv_quasi` : Monomial-NFN Quasi


To test with other augmented dataset, change the scale_low, scale_high and augment_path accordingly. 

* Dataset for ReLU: "experiments/augmented_data/augment_zoo_cifar_{scale_high}"  where scale_high can be chosen from 10, 100, 1000, 10000 (scale_low is always 1). Remember to set dset.filter_activation="relu"

## Classify NFNs

Datasets of SIREN weights trained on the MNIST, FashionMNIST, and CIFAR image datasets, available from [this link](https://drive.google.com/drive/folders/15CdOTPWHqDcS4SwbIdm100rXkTYZPcC5?usp=sharing). Download these datasets into `./experiments/data` and untar them:
```sh
tar -xvf siren_mnist_wts.tar  # creates a folder siren_mnist_wts/
tar -xvf siren_fashion_wts.tar  # creates a folder siren_fashion_wts/
tar -xvf siren_cifar_wts.tar  # creates a folder siren_cifar_wts/
```


### CIFAR: 
```
python -m experiments.launch_classify_siren +setup=cifar_hnps_quasi warmup_steps=20000
```

### MNIST: 
```
python -m experiments.launch_classify_siren +setup=mnist_hnps_quasi warmup_steps=10000
```

### Fashion MNIST: 
```
python -m experiments.launch_classify_siren +setup=fashion_hnps_quasi warmup_steps=20000
```


choices for +setup:  {dataset}_{model} 
dataset: `cifar` , `mnist`, `fashion`

model: 
* `mlp`: MLP 
* `np`: NP
* `hnp` : HNP
* `hnps`: Monomial-NFN
* `hnps_quasi`: Monomial-NFN Quasi
