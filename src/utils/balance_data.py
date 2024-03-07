# balancing the values of target value
from src.components.get_data import read_params, get_data
import argparse
from imblearn.over_sampling import SMOTE

def balance_data(df,config_path):
    # balancing data
    config = read_params(config_path)
    smote = SMOTE(sampling_strategy='minority')
    target_col = config['base']['target_col']
    df1, df1[target_col] = smote.fit_resample(df.drop(target_col,axis=1),df[target_col])
    return df1

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config",default="params.yaml")
    parsed_args = args.parse_args()
    balance_data(config_path=parsed_args.config)