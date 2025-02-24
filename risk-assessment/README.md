# Water Damage Estimation Tool

### Overview

This repository provides a Python-based tool for estimating direct water damage to buildings, particularly residential structures, caused by flooding. It includes two implementations of a WaterDamageEstimator class to calculate monetary damage based on water depth and building characteristics.

### Purpose

The tool is designed to:

- Estimate financial damage (€) to buildings from water intrusion.
- Model damage based on water depth, building area, and structural features.
- Support flood impact assessments for residential properties.

### Key Components

Building Class: Represents a building with surface area, outer wall length, and value per square meter.
WaterDamageEstimator Class: Calculates damage using water depth inside the building, a threshold height, and a linear damage factor.
Features: Simple model with threshold-based water entry and damage scaling up to a maximum depth.

### Output

A monetary value (€) representing the estimated direct damage to the building.

### Notes

Version 1 is simpler and assumes linear damage growth; Version 2 offers a more nuanced probabilistic model.
Customize parameters (e.g., threshold height, damage per m²) based on specific use cases.
Ensure data files are present in ./data/WSS/ for Version 1 if using repair/damage lookups.
