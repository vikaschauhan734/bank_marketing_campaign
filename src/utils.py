import os
import sys
from src.exception import CustomException
import dill
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_models(X_train,y_train,X_test,y_test, models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            # beta = 2, because emphaisising 
            train_model_score = fbeta_score(y_train, y_train_pred,beta=2)
            test_model_score = fbeta_score(y_test, y_test_pred,beta=2)
            report[list(models.keys())[i]] = test_model_score
        return report
    except Exception as e:
        raise CustomException(e, sys)