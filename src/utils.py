import os
import sys
from src.exception import CustomException
import json
import dill
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score,accuracy_score, precision_score,recall_score,f1_score
from src.logger import logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as f:
            return dill.load(f)
    except Exception as e:
        raise CustomException(e,sys)
    
def balance_data(array):
    smote = SMOTE(sampling_strategy='minority')
    X, y = smote.fit_resample(array[:,:-1],array[:,-1])
    return np.concatenate((X,y.reshape(-1,1)), axis=1)
    

def evaluate_models(X_train,y_train,X_test,y_test, models, params):
    try:
        report = {}
        score = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            param=params[(list(models.keys()))[i]]
            gs = GridSearchCV(model,param,cv=5)
            gs.fit(X_train, y_train)
            para = {}
            para['params'] = gs.cv_results_['params']
            para['mean_test_score'] = gs.cv_results_['mean_test_score']
            para['rank_test_score'] = gs.cv_results_['rank_test_score']

            logging.info(f"{model}: {para}")

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_acc = accuracy_score(y_train,y_train_pred)
            train_pre = precision_score(y_train, y_train_pred)
            train_re = recall_score(y_train, y_train_pred)
            train_f1 = f1_score(y_train,y_train_pred)
            test_acc = accuracy_score(y_test,y_test_pred)
            test_pre = precision_score(y_test, y_test_pred)
            test_re = recall_score(y_test, y_test_pred)
            test_f1 = f1_score(y_test,y_test_pred)
            # beta = 2, because emphaisizing recall
            train_model_score = fbeta_score(y_train, y_train_pred,beta=2)
            test_model_score = fbeta_score(y_test, y_test_pred,beta=2)
            report[list(models.keys())[i]] = test_model_score
            score[list(models.keys())[i]] = {"params":gs.best_params_,
                                            "train":{"acccuracy":round(train_acc,5),"precision":round(train_pre,5),"recall":round(train_re,5),"f1":round(train_f1,5),"fbeta":round(train_model_score,5)},
                                             "test":{"acccuracy":round(test_acc,5),"precision":round(test_pre,5),"recall":round(test_re,5),"f1":round(test_f1,5),"fbeta":round(test_model_score,5)}}
        score_file = os.path.join("artifacts","score.json")
        with open(score_file,"w") as f:
            json.dump(score, f, indent=4)
        return report
    except Exception as e:
        raise CustomException(e, sys)