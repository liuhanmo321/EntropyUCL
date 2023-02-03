
# Effective Data Selection and Replay for Unsupervised Continual Learning

## Prerequisites
```
$ pip install -r requirements.txt
```

## Run

Please call the run.py to run the experiments. Please check the following examples of running Finetune model. 

Besides using the configure file,  __method__ directly specifies the continual learning to use and __device__ specifies the gpu to run.

__Example Running Code__

* __Split CIFAR-10__ experiments with SimSiam
```
$ python run.py --data_dir ../Data/ --log_dir ./logs/ -c configs/simsiam_c10.yaml --ckpt_dir ./checkpoints/cifar10_results/ --hide_progress --device=0 --method=finetune
```

* __Split CIFAR-100__ experiments with SimSiam

```
$ python run.py --data_dir ../Data/ --log_dir ../logs/ -c configs/simsiam_c100.yaml --ckpt_dir ./checkpoints/cifar100_results/ --hide_progress --device=0 --method=finetune
```

For detailed experiment settings, please check the *.yaml files in the ./configs folder.



The important parameters in *.yaml configure files are:

``` 
    dataset.name: 'seq-cifar10' for CIFAR-10 data set; 'seq-cifar-100' for CIFAR-100 data set.

    model.name: 'simsiam' for using Contrastive Self-Supervised Loss of SimSiam; 'barlowtwins' for using BarlowTwins.

    model.cl_model: The continual learning method. 'cassle_uniform' for ours; 'cassle' for CaSSLe; 'mixup' for LUMP; 'si' for SI' 'der' for DER; 'finetune' for Finetune; 'joint' for Multitask. It will be overwritten when --method is called.

    model.buffer_size: The number of data to store. For LUMP and DER, this is the total memory size; for ours, this is the total memory size divided by the number of data sets to learn.

    train.alpha: The hyper-parameter for baseline methods SI, DER and LUMP, please check [LUMP](https://github.com/divyam3897/UCL) for details.

    train.cluster_type: How to store data. 'pca' for our high entropy method; 'random' for random data; 'k-means' to store k-means cluster centers; 'minvar' to use Min-Var; 'uniform' to store distant data.

    train.add_noise: decide whether to add noise when learning old data

    train.knn_n: the number of knn neighbors when calculating the noise magnitude. '100' for CIFAR-10 and '10' for CIFAR-100.

    train.cluster_number: used in 'minvar', number of clusters to form, set to the same number as the classes of the input data set.
``` 

<!-- * __Split CIFAR-10__ experiment with BarlowTwins
```
$ python run.py --data_dir ../Data/ --log_dir ../logs/ -c configs/barlow_c10.yaml --ckpt_dir ./checkpoints/cifar10_results/ --hide_progress
```

* __Split CIFAR-100__ experiment with BarlowTwins

```
$ python run.py --data_dir ../Data/ --log_dir ../logs/ -c configs/barlowm_c100.yaml --ckpt_dir ./checkpoints/cifar100_results/ --hide_progress
``` -->

<!-- * __Split Tiny-ImageNet__ experiment with BarlowTwins

```
$ python main.py --data_dir ../Data/ --log_dir ../logs/ -c configs/barlowm_tinyimagenet.yaml --ckpt_dir ./checkpoints/tinyimagenet_results/ --hide_progress
``` -->

<!-- ## Contributing
We'd love to accept your contributions to this project. Please feel free to open an issue, or submit a pull request as necessary. If you have implementations of this repository in other ML frameworks, please reach out so we may highlight them here. -->

## Acknowledgment
The code is build upon [aimagelab/mammoth](https://github.com/aimagelab/mammoth), [divyam3897/UCL](https://github.com/divyam3897/UCL) and [PatrickHua/SimSiam](https://github.com/PatrickHua/SimSiam)
