from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def eval_metrics(actual, pred):
    acc = accuracy_score(actual, pred)
    pre = precision_score(actual, pred)
    rec = recall_score(actual, pred)
    f1 = f1_score(actual, pred)
    return acc, pre, rec, f1

if __name__ == "main":
    eval_metrics()