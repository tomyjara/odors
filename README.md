
# Odors Project

## Project Description

This is a PyTorch project for training neural networks. The primary scripts in this project are `run_experiments.py` for running experiments, `model_analysis.py` for analyzing the performance of the models, and `hyperparams.py` for configuring the hyperparameters.

## Installation

First, clone the repository:

```bash
git clone <repository-url>
cd odors_red
```

Then, install the necessary dependencies. This project uses conda for managing dependencies. You can create a new conda environment with all the required packages using the `environment.yml` file:

```bash
conda env create -f environment.yml
```

Activate the new environment:

```bash
conda activate <environment-name>
```

## Running Experiments

To run the experiments, use the `run_experiments.py` script located in the `modules` folder. 

Before running, make sure to set the following variables:

- `EXP_NAME`: The name of your experiment. This is used to name the directory where checkpoints are saved and the final results CSV file.
- `NUMBER_OF_EXPERIMENTS`: The number of experiments to run. Each experiment will use a different set of randomly selected hyperparameters.
- `multilabel_tags`: The tags used for multi-label classification. These should match the labels in your dataset.

You can run the script with:

```bash
python modules/run_experiments.py
```

## Configuring Hyperparameters

The hyperparameters for the models are configured in the `hyperparams.py` script located in the `modules` folder. 

This script provides a function called `select_random_hyperparams`, which generates a dictionary of hyperparameters for a given experiment name. 

Key hyperparameters include model type, learning rate, optimizer, dropout rate, batch size, and the number of hidden layers. 

For more details, refer to the comments in the `hyperparams.py` script.

## Model Analysis

For model analysis, use the `model_analysis.py` script located in the `modules` folder. 

Here are some key variables and settings:

- `SAVE_ACTIVATIONS`: Set to `True` if you want to save the activations of the model.
- `ACTIVATIONS_PATH`: The path where the activations will be saved if `SAVE_ACTIVATIONS` is set to `True`.
- `PREDICTIONS_PATH`: The path where the predictions of the model will be saved.
- `GAT_0`, `GAT_1`, `GAT_2`, `GAT_3`: These are paths to the saved model checkpoints. They can be updated to point to different models for analysis.

You can run the script with:

```bash
python modules/model_analysis.py
```

## Additional Notes

This project is designed to work with a specific set of models and hyperparameters. If you want to use different models or hyperparameters, you will need to modify the respective scripts accordingly. 

If you encounter any issues or have any questions, please open an issue on this repository.
