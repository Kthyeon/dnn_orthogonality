# Paper Title

This repository is the official implementation of "Revisiting Orthogonality Regularization: A Study for Convolutional Neural Networks in Image Classification" for the image classification tasks, published at IEEE Access 2022 Journal ([Paper Link](https://ieeexplore.ieee.org/abstract/document/9804718)).


## Overview
Experiments in our paper can be reproduced as the following .sh files.
To train a vanilla model, firstly run this command:
```train
bash run_vanilla.sh
```
You can get other baseline models by changing `--model` argument. Also, you can also get the baseline model on CIFAR10 by changing `--dataset` arguments to `cifar10, cifar100, imagenet`.  To give an orthogonality constraint,  you can change `--ortho=none` argument such as `srip, ort,ortho,sin_srip,isometry`.  You can find this in `./regularization/orthogonal_weight.py`. Besides, you may further rescale the lambda value via the argument `--lamb_list`. `--lamb_list` contains four values, and they indicates the lambda for different layers: convolutionaly filter with kernel size 3, pointwise filter, depthwise filter, fully connected layer. `--wd_ablation` implies the switch of weight decay. Precisely, if you give other regularizations to certain layer, then weight decay will not be applied when `--wd_ablation` turns on. Furthermore, we add lots of options for experiments such as initialization (`--init`), optimization (`--optim`), scheduler (`--lr_sch`). Even, we add SOTA algorithms for image classification such as label smoothing, mix up, and cutmix.

## Dependencies

Current code base is tested under following environment:
1.  Python 3.7.3
2.  PyTorch 1.6.0
3.  torchvision 0.7.0
4.  Numpy 1.19.2
5.  Pandas 1.2.0
6.  tqdm


## Results
You can reproduce all results in the paper with our code. All results have been described in our paper including Appendix. The results of our experiments are so numerous that it is difficult to post everything here. However, if you experiment several times by modifying the hyperparameter value in the .sh file, you will be able to reproduce all of our analysis.
