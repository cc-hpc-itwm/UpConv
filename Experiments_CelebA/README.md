# Experiments CelebA

This folder contains the following files:
 <ul>
  <li>Visulaization script</li>
  <li>Spectral regularization scripts: train_spectrum.py and module_spectrum.py.</li>
  <li>Link to download [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)</li>
</ul> 

## Training vanilla models
We train different GAN models using this [repo] (https://github.com/LynnHo/DCGAN-LSGAN-WGAN-GP-DRAGAN-Pytorch). Then, we employ our Visualization script to analyse the frequency behaviour.

<img align="center" src="/imgs/1000_celeba.png" width="800"/>

## Training Spectral Regularization models
From the vanilla models, substitue train.py for train_spectrum.py and module.py for module_spectrum.py.
<img align="center" src="/imgs/1000_spectral.png" width="800"/>
