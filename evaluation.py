"""
Code provided with the paper:
"Attribute Prediction as Multiple Instance Learning"
Model evaluation script
22-09-2022
"""

import random
import numpy as np
import torch
import pandas as pd
import os
import argparse

from dataset import CUB, SUN, AWA2
from training import train

if __name__ == '__main__':
    ## Configure evaluation parameters
    # Default input variables
    batch_size = 16
    seed = 0
    dataset_name = 'CUB'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_root = ''

    folder_path = 'models' # Defines the folder with models to be evaluated. Evaluation csv will be output here.

    # Initiate the parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--dataroot", "-dr", help="Set folder containing dataset")
    parser.add_argument("--data", "-d", help="Set dataset")
    parser.add_argument("--batch_size", "-b", help="Set batch size")
    parser.add_argument("--seed", "-s", help="Set seed")
    parser.add_argument("--gpu", "-g", help="Set gpu")
    parser.add_argument("--folder", "-f", help="Set source model folder")

    # Read arguments from the command line
    args = parser.parse_args()

    # Check for -args
    if args.dataroot: data_root = os.path.join(args.dataroot, '')
    if args.data: dataset_name = args.data
    if args.batch_size: batch_size = args.batch_size
    if args.seed: seed = args.seed
    if args.gpu: device = torch.device('cuda:' + args.gpu)
    if args.folder: folder_path = args.folder

    ## Setup
    # Setup seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    # Setup dataset
    if dataset_name == 'CUB':
        full_dataset = CUB(root_path=data_root)
    elif dataset_name == 'SUN':
        full_dataset = SUN(root_path=data_root)
    elif dataset_name == 'AWA2':
        full_dataset = AWA2(root_path=data_root)

    # Define result dataframe
    result_df = pd.DataFrame(columns=[
        'Model', 's_ap', 'u_ap', 'h_ap',
        's_auc', 'u_auc', 'h_auc',
        'ap_test_per_class', 'auc_test_per_class',
        's_acc_DAP', 'u_acc_DAP', 'h_acc_DAP', 'ZSL_acc_DAP',
        's_acc_il_DAP', 'u_acc_il_DAP', 'h_acc_il_DAP', 'ZSL_acc_il_DAP',
        's_acc_MIL_DAP', 'u_acc_MIL_DAP', 'h_acc_MIL_DAP', 'ZSL_acc_MIL_DAP',
        's_acc_sq_MIL_DAP', 'u_acc_sq_MIL_DAP', 'h_acc_sq_MIL_DAP', 'ZSL_acc_sq_MIL_DAP',
        'Bootstrap_dilution', 'Decorrelation_weight', 'GCE Q'
    ])

    # Loop over models in folder and add results to dataframe
    for file in os.listdir(folder_path):
        if (file[-3:] == '.pt'):  # Check if the file is a saved model
            model_name = file[:-3]
            model_path = os.path.join(folder_path, file)
            result_df = train(full_dataset, device=device, seed=seed, batch_size=batch_size, save_path=folder_path, result_df=result_df,
                              model_name=model_name, do_warm_start = True, model_name_init=model_path,
                              do_save_model=False, do_only_test=True, results_name = 'eval.csv')