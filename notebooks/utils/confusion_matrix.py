from sklearn.metrics import confusion_matrix

def get_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = 0 if tn + fp + fn + tp == 0 else (tp + tn) / (tn + fp + fn + tp)
    recall = 0 if tp + fn == 0 else tp / (tp + fn)
    precision = 0 if tp + fp == 0 else tp / (tp + fp)
    f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    
    return cm, (tn,fp,fn,tp), (accuracy, recall, precision, f1)