# classification metrics
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             matthews_corrcoef, f1_score, confusion_matrix)

# regression metrics
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error)
import pandas as pd

def calculateClassificationMetrics(
        y_true=None, 
        y_pred=None, 
        averge="weighted",
        normalized_cm="true"):


    dict_metrics = {
        "Accuracy" : accuracy_score(y_pred=y_pred, y_true=y_true),
        "Precision" : precision_score(y_true=y_true, y_pred=y_pred, average=averge),
        "Recall" : recall_score(y_true=y_true, y_pred=y_pred, average=averge),
        "F1-score" : f1_score(y_true=y_true, y_pred=y_pred, average=averge),
        "MCC" : matthews_corrcoef(y_true=y_true, y_pred=y_pred),
        "Confusion Matrix" : confusion_matrix(y_true=y_true, y_pred=y_pred, normalize=normalized_cm).tolist()
    }

    return dict_metrics

def calculateRegressionMetrics(
        y_true=None, 
        y_pred=None):
    
    df_values = pd.DataFrame()
    df_values["y_true"] = y_true
    df_values["y_pred"] = y_pred

    dict_metrics = {
        "R2" : r2_score(y_true=y_true, y_pred=y_pred),
        "MAE" : mean_absolute_error(y_true=y_true, y_pred=y_pred),
        "MSE" : mean_squared_error(y_true=y_true, y_pred=y_pred),
        "Kendall-tau" : df_values.corr(method="kendall")["y_true"]["y_pred"],
        "Pearson" : df_values.corr(method="pearson")["y_true"]["y_pred"],
        "Spearman" : df_values.corr(method="spearman")["y_true"]["y_pred"]
    }

    return dict_metrics
