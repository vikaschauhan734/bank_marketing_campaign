# load the train and test files
# train the model
# save the metrics, params
# save the model
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.components.get_data import read_params
from src.utils.eval_metrics import eval_metrics
import pickle
import os
import json

def train_and_evaluate(config_path):
    config = read_params(config_path)
    test_data_path = config['split_data']['test_path']
    train_data_path = config['split_data']['train_path']
    random_state = config['base']['random_state']
    model_dir = config['model_dir']
    params_file = config['reports']['params']
    scores_file = config['reports']['scores']
    n_estimators = config['estimators']['RandomForestClassifier']['params']['n_estimators']
    max_depth = config['estimators']['RandomForestClassifier']['params']['max_depth']
    min_samples_split = config['estimators']['RandomForestClassifier']['params']['min_samples_split']
    min_samples_leaf = config['estimators']['RandomForestClassifier']['params']['min_samples_leaf']

    target = [config['base']['target_col']]

    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")

    y_train = train[target]
    y_test = test[target]

    X_train = train.drop(target, axis=1)
    X_test = test.drop(target, axis=1)

    model = RandomForestClassifier(n_estimators=n_estimators,
                                   max_depth=max_depth,
                                   min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf,
                                   random_state=random_state)
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    (acc, pre, rec, f1) = eval_metrics(y_test,y_pred)
    print(f"Random forest classifier model: n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}")
    print(f"Accuracy score: {acc}")
    print(f"Precision score: {pre}")
    print(f"Recall score: {rec}")
    print(f"f1 score: {f1}")

    with open(scores_file, "a") as f:
        scores = {
            "accuracy": acc,
            "precision": pre,
            "recall": rec,
            "f1": f1
        }
        json.dump(scores, f, indent=4)

    with open(params_file, "a") as f:
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf
        }
        json.dump(params, f, indent=4)


    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir,"model.pkl"), 'wb') as f:
        pickle.dump(model,f)


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)