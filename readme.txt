Code provided with the paper:
"Attribute Prediction as Multiple Instance Learning"
22-09-2022

######################################################################################################################
Summary

This folder contains the python files listed below. They are run on one of  the three datasets: CUB-200-2011, 
SUNAttributes and AWA2, which are not included with this directory. The usage of the scripts is further described in 
the next sections.

Scripts:
- main.py:          Script for main experiments that saves models
- hpsearch.py:      Script for hyperparameter search with a validation set that only saves results
- evaluation.py     Script for evaluation existing models that only saves results

Other python files:
- training.py       Function with main training loop
- dataset.py        Dataset classes
- network.py        Network and loss classes
- util.py           Utility functions

######################################################################################################################
main.py

Script with the main experiments that are tested against the test set. Creates a folder called results and within it a
 subfolder with the name of the experiment. In this subfolder it generates csv file with the experiment results
 and a number of saved model weights.

Mandatory arguments are:
--name:             Experiment name, used to store models and results
--data:             'CUB'/'SUN'/'AWA2' dataset used
--dataroot:         Folder containing dataset subfolder.

Other arguments:
--batch_size:       batch size, default 16
--seed:             seed for random functions, default 0
--gpu:              id of gpu used for cuda, default 0 if a gpu is available

Example:
python3 main.py --name CUB01 --data CUB --dataroot '../data'

######################################################################################################################
hpsearch.py

Script with hyperparameter search on validation set. Arguments are the same as main.py. Runs a simple hyperparameter
search and evaluates the results against a validation set. Models are not saved but results are stored in subfolder.

Example:
python3 hpsearch.py --name SUN_hp_01 --data SUN --dataroot '../data'
python3 hpsearch.py --name SUN_hp_01 --data SUN --dataroot '../data' --batch_size 32 --seed 16 --gpu 1

######################################################################################################################
evaluation.py

Script that loops over a set of models in a folder and evaluates them against the test set. Results are saved in
eval.csv in the folder containing the models. Arguments similar to main.py, but the --name argument is replaced with
the folder that stores the trained models.

Mandatory arguments are:
--folder:           path to folder with trained models. Output csv is stored here.
--data:             'CUB'/'SUN'/'AWA2' dataset used
--dataroot:         Folder containing dataset subfolder.

Other arguments:
--batch_size:       batch size, default 16
--seed:             seed for random functions, default 0
--gpu:              id of gpu used for cuda, default 0 if a gpu is available

Example:
python3 evaluation.py --folder 'results/CUB01' --data CUB --dataroot '../data'
