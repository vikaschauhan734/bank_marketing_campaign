# read the data from data source
# clean the data
# save it in the data/raw for further process
from get_data import read_params, get_data
from src.utils.balance_data import balance_data
from src.utils.data_concat import data_concat
from src.utils.clean_col import clean_col
import argparse

def load_and_save(config_path):
    config = read_params(config_path)
    df = get_data(config_path)
    # drop duplicate values
    df.drop_duplicates(inplace=True)
    # clean categorical data
    for col in df.columns:
        if df[col].dtypes =="O":
            df[col] = clean_col(df,col)
    cat_col = [col for col in df.columns if df[col].dtypes == "O" ]
    df = data_concat(df,cat_col)
    new_cols=[col for col in df.columns]
    raw_data_path = config['load_data']['raw_dataset_csv']
    df = balance_data(df,config_path)
    df.to_csv(raw_data_path,sep=",",index=False,header=new_cols)
    

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config",default="params.yaml")
    parsed_args = args.parse_args()
    load_and_save(config_path=parsed_args.config)