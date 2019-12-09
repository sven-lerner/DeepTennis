# DeepTennis

CS 230 Tennis Match Prediction Project

This project implements an LSTM recurrent neural network to predict tennis match win probabilities using sequential point-by-point data of the four major tennis tournaments. Data is courtesy of Jeff Sackman. Model was found to produce 79.5% accuracy across all points on a test set of matches from 2014.

## Section Descriptions

### Data

All datasets used for the project. final_data and gollubdata contain the final, cleaned, and augmented datasets used for point-by-point and pre-match data respectively.

### Data Loaders

Python scripts written to download raw data and produce final data outputs. Contains preprocessing scripts to fill missing data, pair point-by-point data to pre-match predictions,  and introduce additional fields. 

### Models

Contains model infrastructure file and loss function helpers. Used to create a model instance that is later trained/tested on.

### Notebooks

Data exploration, debugging, and results analysis notebooks.

### Saved Models

Cached model instances to reproduce important results.

### environment.yml

Virtual environment used to run model. Recommended for future use of this code.

### test_model.py and train.py

Main scripts to create a model instance, load data, and train or test the model respectively. Model building functionality largely handled by model scripts previously mentioned, while hyperparameter search and tweaking performed via train file. 
