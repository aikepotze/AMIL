"""
Code provided with the paper:
"Attribute Prediction as Multiple Instance Learning"
Main experiment script
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
    ## Configure experiment parameters
    # Default input variables
    batch_size = 16
    seed = 0
    dataset_name = 'CUB'  # CUB, SUN or AWA
    experiment_name = 'CUB01'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_root = ''

    # Initiate the parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--name", "-n", help="Set experiment name")
    parser.add_argument("--dataroot", "-dr", help="Set folder containing dataset")
    parser.add_argument("--data", "-d", help="Set dataset")
    parser.add_argument("--batch_size", "-b", help="Set batch size")
    parser.add_argument("--seed", "-s", help="Set seed")
    parser.add_argument("--gpu", "-g", help="Set gpu")

    # Read arguments from the command line
    args = parser.parse_args()

    # Check for -args
    if args.name: experiment_name = args.name
    if args.data: dataset_name = args.data
    if args.batch_size: batch_size = int(args.batch_size)
    if args.seed: seed = int(args.seed)
    if args.gpu: device = torch.device('cuda:' + args.gpu)
    if args.dataroot: data_root = os.path.join(args.dataroot, '')

    ## Setup
    # Setup seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Setup dataset and dataset-specific hyperparemeters
    if dataset_name == 'CUB':
        full_dataset = CUB(root_path=data_root)
        decorrelation_weight = 10.0
    elif dataset_name == 'SUN':
        full_dataset = SUN(root_path=data_root)
        decorrelation_weight = 5.0
    elif dataset_name == 'AWA2':
        full_dataset = AWA2(root_path=data_root, samples_per_class=500)
        decorrelation_weight = 5.0

    # Setup output path and results dataframe
    save_path = os.path.join('results/' + experiment_name)
    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        raise Exception('Experiment directory already exists!')

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

    ## Experiments
    # Image-level gt baseline
    if dataset_name != 'AWA2':
        result_df = train(full_dataset, device=device, seed=seed, batch_size=batch_size, save_path=save_path, result_df=result_df,
                          model_name='Image_level',
                          class_level_gt=False)

    ## Without decorellation
    # Class-level baseline
    result_df = train(full_dataset, device=device, seed=seed, batch_size=batch_size, save_path=save_path, result_df=result_df,
                      model_name='Class_level')

    # Deep-MML baseline
    result_df = train(full_dataset, device=device, seed=seed, batch_size=batch_size, save_path=save_path,
                      result_df=result_df,
                      do_mil_base=True,
                      model_name='Deep_MML')

    #GCE
    result_df = train(full_dataset, device=device, seed=seed, batch_size=batch_size, save_path=save_path, result_df=result_df,
                      model_name='GCE',
                      do_truncated_loss=True,
                      truncated_loss_q=0.3)
    #GCE MIL
    result_df = train(full_dataset, device=device, seed=seed, batch_size=batch_size, save_path=save_path,
                      result_df=result_df,
                      model_name='GCE_MIL',
                      do_truncated_loss=True,
                      apply_only_to_positives=True,
                      truncated_loss_q=0.5)

    # Bootstrap baseline
    result_df = train(full_dataset, device=device, seed=seed, batch_size=batch_size, save_path=save_path, result_df=result_df,
                      model_name='Boot',
                      dilute_predictions=0.8)

    # MIL Bootstrap
    result_df = train(full_dataset, device=device, seed=seed, batch_size=batch_size, save_path=save_path, result_df=result_df,
                      model_name='MIL_Boot',
                      apply_only_to_positives=True,
                      dilute_predictions=0.8)


    ## With decorrelation
    # Class-level baseline
    result_df = train(full_dataset, device=device, seed=seed, batch_size=batch_size, save_path=save_path,
                      result_df=result_df,
                      model_name='Class_level_G',
                      decorrelation_weight=decorrelation_weight)

    # Deep-MML baseline
    result_df = train(full_dataset, device=device, seed=seed, batch_size=batch_size, save_path=save_path,
                      result_df=result_df,
                      do_mil_base=True,
                      model_name='Deep_MML_G',
                      decorrelation_weight=decorrelation_weight)

    # GCE
    result_df = train(full_dataset, device=device, seed=seed, batch_size=batch_size, save_path=save_path,
                      result_df=result_df,
                      model_name='GCE_G',
                      do_truncated_loss=True,
                      truncated_loss_q=0.3,
                      decorrelation_weight=decorrelation_weight)
    # GCE MIL
    result_df = train(full_dataset, device=device, seed=seed, batch_size=batch_size, save_path=save_path,
                      result_df=result_df,
                      model_name='GCE_MIL_G',
                      do_truncated_loss=True,
                      apply_only_to_positives=True,
                      truncated_loss_q=0.5,
                      decorrelation_weight=decorrelation_weight)


    # Bootstrap + decorrelation
    result_df = train(full_dataset, device=device, seed=seed, batch_size=batch_size, save_path=save_path, result_df=result_df,
                      model_name='Boot_G',
                      dilute_predictions=0.8,
                      decorrelation_weight=decorrelation_weight)

    # MIL Bootstrap + decorrelation
    result_df = train(full_dataset, device=device, seed=seed, batch_size=batch_size, save_path=save_path, result_df=result_df,
                      model_name='MIL_Boot_G',
                      apply_only_to_positives=True,
                      dilute_predictions=0.8,
                      decorrelation_weight=decorrelation_weight)
