## Getting Started

Follow these steps to set up the project and get started with training or testing.

### 1️) Create and Activate a Virtual Environment

First, create a virtual environment in a cmd terminal. If you're using Python 3, run:

```bash
python -m venv .venv_evtGAN
```

After creating, activate the virtual environment:

```bash
.venv\Scripts\Activate
```

### 2) Install Required Packages

After activating the virtual environment, install the required dependencies (this may take a while) using:

```bash
pip install -r requirements.txt
```

### 3) Running Training or Testing Setup

To run the training setup::

```bash
python train.py
```

Check saved_models/<timestamp>/ for generator_weights.pth, config.yaml, and generator.pt.

```bash
python generate.py --model_dir <timestamp> --model_type pickle --n_gen 5000
python generate.py --model_dir <timestamp> --model_type torchscript
```

# Model Architecture

#### Generator

![Gen](master-thesis/evtGAN/images/Generator.jpg)

#### Discriminator

![Dis](master-thesis/evtGAN/images/Discriminator.jpg)

#### File Descriptions

- config/config.yaml: Stores all hyperparameters (e.g., noise dimension, epochs) and file paths.
- data/precipitation_maxima.nc: The input NetCDF file containing training data (e.g., precipitation maxima). You’ll need to provide this file.
- data/synthetic/: Output folder where generated .csv files will be saved (created automatically).
- models/evtgan.py: Defines the GAN’s Generator and Discriminator classes.
- utils/data_utils.py: Loads and preprocesses .nc data (e.g., transforming to uniform margins, padding).
- utils/evt_utils.py: Fits Generalized Extreme Value (GEV) distributions and transforms generated samples back to the original scale.
- train.py: Trains the GAN and saves the trained generator.
- generate.py: Generates synthetic samples using the trained model and saves them as .csv.
- requirements.txt: Lists required Python packages.

#### Using Synthetic Samples

The synthetic precipitation data can be used for:

- Input to hydrological models for flood event simulation.
- Statistical analysis of extreme event frequency and intensity.
- Training other machine learning models for prediction tasks.
