

## Abstract


__Contribution of this work__


## Prerequisites
```
$ pip install -r requirements.txt
```

## Run
* __Split CIFAR-10__ experiment with SimSiam
```
$ python main.py --data_dir ../Data/ --log_dir ../logs/ -c configs/simsiam_c10.yaml --ckpt_dir ./checkpoints/cifar10_results/ --hide_progress
```

* __Split CIFAR-100__ experiment with SimSiam

```
$ python main.py --data_dir ../Data/ --log_dir ../logs/ -c configs/simsiam_c100.yaml --ckpt_dir ./checkpoints/cifar100_results/ --hide_progress
```

* __Split Tiny-ImageNet__ experiment with SimSiam

```
$ python main.py --data_dir ../Data/ --log_dir ../logs/ -c configs/simsiam_tinyimagenet.yaml --ckpt_dir ./checkpoints/tinyimagenet_results/ --hide_progress
```

* __Split CIFAR-10__ experiment with BarlowTwins
```
$ python main.py --data_dir ../Data/ --log_dir ../logs/ -c configs/barlow_c10.yaml --ckpt_dir ./checkpoints/cifar10_results/ --hide_progress
```

* __Split CIFAR-100__ experiment with BarlowTwins

```
$ python main.py --data_dir ../Data/ --log_dir ../logs/ -c configs/barlowm_c100.yaml --ckpt_dir ./checkpoints/cifar100_results/ --hide_progress
```

* __Split Tiny-ImageNet__ experiment with BarlowTwins

```
$ python main.py --data_dir ../Data/ --log_dir ../logs/ -c configs/barlowm_tinyimagenet.yaml --ckpt_dir ./checkpoints/tinyimagenet_results/ --hide_progress
```

<!-- ## Contributing
We'd love to accept your contributions to this project. Please feel free to open an issue, or submit a pull request as necessary. If you have implementations of this repository in other ML frameworks, please reach out so we may highlight them here. -->

## Acknowledgment
The code is build upon [aimagelab/mammoth](https://github.com/aimagelab/mammoth), LUMP and [PatrickHua/SimSiam](https://github.com/PatrickHua/SimSiam)

## Citation
If you found the provided code useful, please cite our work.

```bibtex

```
