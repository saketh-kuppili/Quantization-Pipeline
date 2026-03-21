from sklearn.metrics import accuracy_score

def compute_accuracy(preds, labels):
    return accuracy_score(labels, preds)