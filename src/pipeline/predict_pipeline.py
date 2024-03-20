import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = os.path.join('artifacts','model.pkl')
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled[:,:-1])
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 age: int,
                 job: str,
                 marital: str,
                 education: str,
                 default: str,
                 housing: str,
                 loan: str,
                 contact: str,
                 month: str,
                 day_of_week: str,
                 duration: int,
                 campaign: int,
                 pdays: int,
                 previous: int,
                 poutcome: str,
                 emp_var_rate: float,
                 cons_price_idx: float,
                 cons_conf_idx: float,
                 euribor3m: float,
                 nr_employed: float):
        self.age = age
        self.job = job
        self.marital = marital
        self.education = education
        self.default = default
        self.housing = housing
        self.loan = loan
        self.contact = contact
        self.month = month
        self.day_of_week = day_of_week
        self.duration = duration
        self.campaign = campaign
        self.pdays = pdays
        self.previous = previous
        self.poutcome = poutcome
        self.emp_var_rate = emp_var_rate
        self.cons_price_idx = cons_price_idx
        self.cons_conf_idx = cons_conf_idx
        self.euribor3m = euribor3m
        self.nr_employed = nr_employed

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "age": [self.age],
                "job": [self.job],
                "marital": [self.marital],
                "education": [self.education],
                "default": [self.default],
                "housing": [self.housing],
                "loan": [self.loan],
                "contact": [self.contact],
                "month": [self.month],
                "day_of_week": [self.day_of_week],
                "duration": [self.duration],
                "campaign": [self.campaign],
                "pdays": [self.pdays],
                "previous": [self.previous],
                "poutcome": [self.poutcome],
                "emp.var.rate": [self.emp_var_rate],
                "cons.price.idx": [self.cons_price_idx],
                "cons.conf.idx": [self.cons_conf_idx],
                "euribor3m": [self.euribor3m],
                "nr.employed": [self.nr_employed],
                "y":["no"]
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)


        