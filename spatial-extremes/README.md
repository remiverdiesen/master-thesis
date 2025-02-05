# Spatial Extremes

Models complex spatial dependencies between climate extremes in different locations. With the evtGAN, we show that combining extreme value theory (evt) with a deep learning model (generative adversarial networks (GAN)) can well represent complex spatial dependencies between precipitation extremes. Hence, instead of running expensive climate models, the approach can be used to sample many instances of spatially cooccurring extremes with realistic dependence structure, which may be used for climate risk modeling and stress testing of climate-sensitive systems.

Source: Application Paper: Boulaguiem et al. (2022) - Modeling and simulating spatial extremes by combining extreme value theory with generative adversarial networks.

### Architecture

![Architecture](docs/images/Discriminator.jpg)
![Architecture](docs/images/Generator.jpg)

## 🚀 Getting Started

Follow these steps to set up the project and get started with training or testing.

### 1️) Create and Activate a Virtual Environment

First, create a virtual environment. If you're using Python 3, run:

```bash
python -m venv .venv
```

### 2) Install Required Packages

After activating the virtual environment, install the required dependencies using:

```bash
pip install -r requirements.txt
```

### 3) Running Training or Testing Setup

To run the training setup::

```bash
python train.py
```
