# [Watch your Up-Convolution: CNN Based Generative Deep Neural Networks are Failing to Reproduce Spectral Distributions](https://arxiv.org/abs/1911.00686)

This repository provides the official Python implementation of Watch your Up-Convolution: CNN Based Generative Deep Neural Networks are Failing to Reproduce Spectral Distributions (Paper: [https://arxiv.org/abs/1911.00686](https://arxiv.org/abs/1911.00686)).

<img align="center" src="imgs/pipeline.png" width="1000"/>

Overview of the pipeline used in our approach. It contains two main blocks, a pre-processing where the input is transformed to a more convenient domain and a training block, where a classifier uses the new transformed features to determine whether the face is real or not. Notice that input images are grey-scaled
before DFT.


## Dependencies
Tested on Python 3.6.x.
* [Pytorch](https://pytorch.org/get-started/previous-versions/) (1.1.0)
* [NumPy](http://www.numpy.org/) (1.16.2)
* [Opencv](https://opencv.org/opencv-4-0/) (4.0.0)
* [Matplotlib](https://matplotlib.org/) (3.1.1)



###  Citation
If this work is useful for your research, please cite our [paper](https://arxiv.org/abs/1911.00686):
```
@misc{durall2019unmasking,
    title={Unmasking DeepFakes with simple Features},
    author={Ricard Durall and Margret Keuper and Franz-Josef Pfreundt and Janis Keuper},
    year={2019},
    eprint={1911.00686},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
