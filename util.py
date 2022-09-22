"""
Code provided with the paper:
"Attribute Prediction as Multiple Instance Learning"
Utility functions
22-09-2022
"""


import pandas as pd
import os

def save(result_df, run_metrics, save_path, name):
    """Appends vector 'run_metrics' to pandas dataframe 'result_df' and saves to 'save_path'."""

    result_df = result_df.append(pd.Series(run_metrics, index=result_df.columns), ignore_index=True)
    result_df.to_csv(os.path.join(save_path + '/' + name))
    return result_df
