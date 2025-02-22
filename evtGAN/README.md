### Repository Structure

Here’s the folder structure for the evtGAN repository:

evtGAN/
├── config/
│ └── config.yaml # Configuration file for hyperparameters and settings
├── data/
│ ├── precipitation_maxima.nc  
│ ├── ids.nc
│ └── synthetic/ # Output folder for generated CSV files
├── models/
│ └── evtgan.py # Contains Generator and Discriminator classes
├── utils/
│ ├── data_utils.py # Functions for loading and preprocessing NetCDF data
│ └── evt_utils.py # Functions for EVT transformations (GEV fitting, etc.)
├── train.py # Script to train the GAN model
├── generate.py # Script to generate synthetic samples and save as CSV
└── requirements.txt # Python dependencies

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
