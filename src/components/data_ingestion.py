import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from src.utils.clean_col import clean_col
from src.utils.data_concat import data_concat
from src.utils.balance_data import balance_data

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('data_given/bank-additional-full.csv')
            logging.info("Read the dataset as dataframe")

            # drop duplicate values
            df.drop_duplicates(inplace=True)
            # clean categorical data
            for col in df.columns:
                if df[col].dtypes =="O":
                    df[col] = clean_col(df,col)
                    pd.set_option('future.no_silent_downcasting', True)
            cat_col = [col for col in df.columns if df[col].dtypes == "O" ]
            df = data_concat(df,cat_col)
            df = balance_data(df)
            logging.info("Dataset cleaned")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Train test split initiated")

            train_set,test_set = train_test_split(df, test_size=0.2, random_state=42, stratify=df['y'])
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Ingestion of data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()