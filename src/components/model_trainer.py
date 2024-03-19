import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, accuracy_score, f1_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, balance_data, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            train_array = balance_data(train_array)
            test_array = balance_data(test_array)
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Logistic Regression": LogisticRegression(),
                "SVC": SVC(),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "Decision Tree Classifier": DecisionTreeClassifier(),
                "Random Forest Classifier": RandomForestClassifier(),
                "Ada Boost Classifier": AdaBoostClassifier(),
                "XGBClassifier":XGBClassifier(),
                "Gradient Boosting Classifier": GradientBoostingClassifier(),
                "Bagging Classifier": BaggingClassifier(),
                "GaussianNB": GaussianNB()
            }

            params={
                "Logistic Regression": {
                    'penalty':['l2']
                },
                "SVC":{
                    'kernel':['linear','poly','rbf','sigmoid'],
                    'gamma':['scale','auto']
                },
                "K-Neighbors Classifier":{
                    'n_neighbors':[4,5,6,7,10],
                    'weights':['uniform','distance'],
                    'algorithm':['auto','ball_tree','kd_tree','brute']
                },
                "Decision Tree Classifier":{
                    'criterion':['gini','entropy','log_loss'],
                    'splitter':['best','random'],
                    'max_features':['sqrt','log2']
                },
                "Random Forest Classifier":{
                    'n_estimators': [10,25,31,50],
                    'max_depth': [10,13,15],
                    'min_samples_split': [3,4,5],
                    'min_samples_leaf': [5,6,7]
                },
                "Ada Boost Classifier":{
                    'n_estimators':[10,15,20,30,50],
                    'learning_rate':[0.01,0.1,1,2,5,10]
                },
                "XGBClassifier":{
                    'booster':['gbtree','gblinear'],
                    'eta':[0.1,0.3,0.5],
                    'gamma':[0,1,10,50,100],
                    'max_depth':[0,1,5,10]
                },
                "Gradient Boosting Classifier":{
                    'loss':['log_loss','exponential'],
                    'learning_rate':[0.1,0.25,0.5,1],
                    'n_estimators':[10,20,50,75,100],
                    'subsample':[0.1,0.25,0.5,0.75,1],
                    'criterion':['friedman_mse','squared_error'],
                    'min_samples_split':[2,5,10],
                    'min_samples_leaf':[1,2,10]
                },
                "Bagging Classifier":{
                    'n_estimators':[1,5,10,15,20],
                    'max_samples':[0.1,1,10],
                    'max_features':[0.1,1,10]
                },
                "GaussianNB":{

                }

            }

            model_report:dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test, models=models,params=params)
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info("Best model found on the both training and testing dataset")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            cls_report = classification_report(y_test, predicted)
            return cls_report
        except Exception as e:
            raise CustomException(e,sys)