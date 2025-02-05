# Climate Credit Risk modelling

This is the repository of the implementation of the master thesis of Remi Verdiesen, for the Vrije Universiteit and SAS Institute.

### Overview

Three parts:

- Spatial-extremes
- Hydrolic Modelling
- Risk Assessment

Models complex spatial dependencies between climate extremes in different locations. With the evtGAN, we show that combining extreme value theory (evt) with a deep learning model (generative adversarial networks (GAN)) can well represent complex spatial dependencies between precipitation extremes. Hence, instead of running expensive climate models, the approach can be used to sample many instances of spatially cooccurring extremes with realistic dependence structure, which may be used for climate risk modeling and stress testing of climate-sensitive systems.

Source: Application Paper: Boulaguiem et al. (2022) - Modeling and simulating spatial extremes by combining extreme value theory with generative adversarial networks.

### Getting started

![Experiments](docs/images/Experiments.jpg)

### linux command

'history'

### commands lightning

'unset GITHUB_TOKEN'
'git config --global user.name "remiverdiesen"'
'git config --global user.email "remi.verdiesen@hetnet.nl"'
'gh auth login'
'git clone https://github.com/remiverdiesen/spatial-extremes.git'

'ssh-keygen -t ed25519'
'eval "$(ssh-agent -s)"'
'ssh-add ~/.ssh/id_ed25519'
'cat ~/.ssh/id_ed25519.pub'
'ssh -T git@github.com'
