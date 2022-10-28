# MindSpore CSWin-Transformer

## Introduction

This work is used for reproduce CSWin-Transformer based on NPU(Ascend 910)

**CSWin-Transformer** is introduced in [arxiv](https://arxiv.org/abs/2107.00652)

A challenging issue in Transformer design is that global self-attention is very expensive to compute whereas local self-attention often limits the field of interactions of each token.To address this issue, CSWin-Transformer computes self-attention in the horizontal and vertical stripes in parallel that from a cross-shaped window, with each stripe obtained by splitting the input feature into stripes of equal width. With CSWin, global attention with a limited computation cost is realized.

CSWin Transformer achieves strong performance on ImageNet classification (83.6 on val with 15G flops)

![framework](/figures/cswin-teaser.png)

## Data preparation

Download and extract [ImageNet](https://image-net.org/).

The directory structure is the standard layout for the MindSpore [`dataset.ImageFolderDataset`](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/dataset/mindspore.dataset.ImageFolderDataset.html?highlight=imagefolderdataset), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```
## Training


```
mpirun -n 8 python train.py --config <config path> > train.log 2>&1 &
```

## Evaluation 


```
python eval.py --config <config path>
```



## Acknowledgement

We heavily borrow the code from [CSWin-Transformer](https://github.com/microsoft/CSWin-Transformer) and [swin_transformer](https://gitee.com/mindspore/models/tree/master/research/cv/swin_transformer)
We thank the authors for the nicely organized code!
