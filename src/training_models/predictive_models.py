from joblib import dump, load
from sklearn.model_selection import StratifiedKFold, cross_validate

from training_models.performance_models import (
    calculate_classification_metrics,
    calculate_metrics_kfold,
    calculate_regression_metrics,
)


class PredictiveModel(object):
    def __init__(self, X_train=None, X_val=None, y_train=None, y_val=None):
        self.model = None
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val

        self.performances = None

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)

    def train_model_with_kfold(self, k=10, scores=[], stratified=False, preffix=""):
        if not stratified:
            response_cv = cross_validate(
                self.model, self.X_train, self.y_train, cv=k, scoring=scores
            )
        else:
            cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

            response_cv = cross_validate(
                self.model, self.X_train, self.y_train, cv=cv, scoring=scores
            )

        self.model.fit(self.X_train, self.y_train)

        return calculate_metrics_kfold(response_cv, scoring_list=scores, preffix=preffix)

    def eval_model(
        self, y_true, y_pred, type_model="class", averge="weighted", normalized_cm="true"
    ):
        if type_model == "class":
            return calculate_classification_metrics(
                y_true=y_true, y_pred=y_pred, averge=averge, normalized_cm=normalized_cm
            )
        else:
            return calculate_regression_metrics(y_true=y_true, y_pred=y_pred)

    def export_model(self, name_export="trained_model.joblib"):
        dump(self.model, name_export)

    def load_model(self, name_model="trained_model.joblib"):
        self.model = load(name_model)

    def make_predictions_with_model(self, X_matrix=None):
        return self.model.predict(X_matrix)
