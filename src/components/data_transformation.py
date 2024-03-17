import os
import sys
from dataclasses import dataclass
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

def maximum_occurring(df):
    for col in df.columns:
        if df[col].dtypes =="O":
            df[col] = df[col].replace("unknown",df[col].value_counts().idxmax()).infer_objects(copy=False)
    pd.set_option('future.no_silent_downcasting', True)
    return df

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data tranformation
        '''
        try:
            numerical_columns = ['age', 'duration', 'campaign', 'pdays', 'previous',
                                 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
            categorical_columns = ['job', 'marital', 'education', 'default', 'housing',
                                   'loan', 'contact', 'month', 'day_of_week', 'poutcome', 'y']

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            logging.info(f"Numerical columns: {numerical_columns}")
            cat_cleaner = FunctionTransformer(func=maximum_occurring)
            cat_pipeline = Pipeline(
                steps=[
                    ("cat_cleaner",cat_cleaner),
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder(drop='first')),
                    #("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info(f"Categorical columns: {categorical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformtion(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            train_arr = preprocessing_obj.fit_transform(train_df)
            test_arr = preprocessing_obj.transform(test_df)

            logging.info("Saved preprocessing object.")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)