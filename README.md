# Multi-Modality Transfer Learning - Network Implementation

This repository contains a Pytorch implementation of the end-to-end sign language translation network proposed by Yutong Chen Et al., in the paper: [A Simple Multi-Modality Transfer Learning Baseline for Sign Language Translation](https://arxiv.org/abs/2203.04287)[1]. It also contains custom Pytorch datasets for [RWTH-PHOENIX-Weather 2014](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/)[2], [BOBSL](https://www.robots.ox.ac.uk/~vgg/data/bobsl/)[3], [WLASL](https://dxli94.github.io/WLASL/)[4] and [Auslan Corpus](https://www.elararchive.org/dk0001)[5], to allow for easy reproducibility. 

At the time of release, no repository had been released for this paper. There now exists a official version [here](https://github.com/FangyunWei/SLRT/tree/main/TwoStreamNetwork). 

## Description

### Overview
The sign language datasets currently available, such as PHOENIX-2014T and CSL-Daily, only have 10K-20K sets of sign videos, gloss annotations, and texts. This is much smaller than the typical parallel data used to train spoken language translation models, which usually have around 1 million samples. Therefore, the lack of data is a major obstacle in creating accurate sign language translation models. This proposed end-to-end network combined with progressive pretraining is an attempt to solve this problem. The pretraining starts with general-domain datasets and moves towards within-domain datasets. At the time of publication this achieved state of the art on sign language translation. 

### Implementation Notes
The MMTL network is contained within MMTL.py and the custom datasets are implemented in datasets.py. In the hopes of speeding up reproducibility, there are three examples of training the network in: example-Elar, example-Phoenix and example-Wlasl. Due to dataset copyrights, the pre-trained network weights are unable to be released.  

You can however utilize the pre-trained weights for the public datasets. The language transformer, mBart, will auto-load the weights (mBart-large-cc25). If you wish to use the backbone pre-trained on kinetics400, download the weight file from [here](https://github.com/kylemin/S3D), and add it to the root directory named "./S3D_kinetics400.pt" (or simply edit where necessary).

## Getting Started

### Dependencies
``` Python
Pillow
torch
torchvision==0.14.0
torchtext
sentencepiece
torch-position-embedding
webvtt-py
transformers
pytorchvideo
```

### Executing program
1) Clone repo
2) Install Dependancies 
3) Download Phoenix Dataset to root of repo, name folder "PHOENIX-2014-T"
4) Run example-Phoenix.py

### Training On New Languages
To train the network on new languages (e.g. German Sign Language), one needs to do two key things: 
1) Expand the language codes contained within the list FAIRSEQ_LANGUAGE_CODES in customTokenizer.py. To do this, simply come up with a unique code lang code not contained within the list, and hard code it into the end. 

2) After adding a new language token/s, get the entire new languages vocabulary and add it to the language modules tokenizer, then resize the token embeddings layer in the transformer. This can be done in a single step, by calling ```add_tokens``` in either the Language module or the EndToEnd module.


## Author Notes
This repository was built for a 10-week research effort into Auslan sign language translation (SLT). It aimed to answer if pretraining on a dataset with no glosses but a large vocabulary overlap (BOBSL), could improve the performance of Auslan SLT. Unfortunately, because of hardware and time constraints, this network could not be trained on the BOBSL dataset, thus unable to answer the key question. On the positive side, it was able to demonstrate improvements on Auslan SLT by progressive pretraining on cross-domain languages (ASL, DGS) and general-domain tasks (action recognition and language translation). 

If you have any questions or need help in implementation, feel free to PR this README. 

## References
<a id=1>[1]</a> Chen, Yutong, Fangyun Wei, Xiao Sun, Zhirong Wu and Stephen Lin. “A Simple Multi-Modality Transfer Learning Baseline for Sign Language Translation.” 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2022): 5110-5120.

<a id="2">[2]</a> O. Koller, J. Forster, and H. Ney. [Continuous sign language recognition: Towards large vocabulary statistical recognition systems handling multiple signers](https://www-i6.informatik.rwth-aachen.de/publications/download/996/Koller-CVIU-2015.pdf). [Computer Vision and Image Understanding](http://www.journals.elsevier.com/computer-vision-and-image-understanding/), volume 141, pages 108-125, December 2015.

<a id=3>[3]</a> Albanie, S., Varol, G., Momeni, L., Bull, H., Afouras, T., Chowdhury, H., Fox, N., Woll, B., Cooper, R.J., McParland, A., & Zisserman, A. (2021). BBC-Oxford British Sign Language Dataset. ArXiv, abs/2111.03635.

<a id=4>[4]</a> Li, D., Rodriguez-Opazo, C., Yu, X., & Li, H. (2019). Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison. 2020 IEEE Winter Conference on Applications of Computer Vision (WACV), 1448-1458.

<a id=5>[5]</a> Johnston, Trevor. 2008. Auslan Corpus. Endangered Languages Archive. Handle: http://hdl.handle.net/2196/00-0000-0000-0000-D7CF-8. Accessed on Feburary 2022.

