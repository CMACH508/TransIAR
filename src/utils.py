
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve


def auc(y_true, y_pred):
    roc_auc = roc_auc_score(y_true, y_pred)
    return roc_auc

def aupr(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    aupr = metrics.auc(recall, precision)
    return aupr


def cal_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    '''
            pred_0   pred_1
    true_0
    true_1
    '''
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
    print(cm)

    Accuracy = (TP + TN) / (TP + FP + TN + FN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1_score = (2 * Precision * Recall) / (Precision + Recall)
    print("accuracy: {}".format(round(Accuracy*100,2)))
    print("precision: {}".format(round(Precision*100,2)))
    print("recall: {}".format(round(Recall*100,2)))
    print("F1 score: {}".format(round(F1_score*100,2)))

    AUC = auc(y_true, y_prob)
    AUPR = aupr(y_true, y_prob)
    print("AUC: {}".format(round(AUC*100,2)))
    print("AUPR: {}".format(round(AUPR*100,2)))

