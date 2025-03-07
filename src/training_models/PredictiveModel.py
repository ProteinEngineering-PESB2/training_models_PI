from training_models.PerformancesModels import *
from utils.utilsLib import *

from joblib import load, dump
from sklearn.model_selection import cross_validate, StratifiedKFold

class PredictiveModel(object):

    def __init__(
            self, 
            X_train=None,
            X_val=None,
            y_train=None,
            y_val=None):
        
        self.model = None
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val

        self.performances = None
        
    def trainModel(self):

        self.model.fit(self.X_train, self.y_train)
    
    def trainModelWithKFold(self, k=10, scores=[], stratified=False, preffix=""):

        if stratified == False:
            response_cv = cross_validate(
                self.model, 
                self.X_train, 
                self.y_train, 
                cv=k, 
                scoring=scores)
        else:
            cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
            
            response_cv = cross_validate(
                self.model, 
                self.X_train, 
                self.y_train, 
                cv=cv, 
                scoring=scores)
        
        self.model.fit(self.X_train, self.y_train)
        
        return calculateMetricsKFold(response_cv, scoring_list=scores, preffix=preffix)

    def evalModel(
            self,
            y_true,
            y_pred,
            type_model="class",
            averge="weighted",
            normalized_cm="true"):
        
        if type_model == "class":
            return calculateClassificationMetrics(
                y_true=y_true, 
                y_pred=y_pred, 
                averge=averge,
                normalized_cm=normalized_cm
            )
        else:
            return calculateRegressionMetrics(
                y_true=y_true, 
                y_pred=y_pred
            )

    def exportModel(
            self, 
            name_export="trained_model.joblib"):
        
        dump(
            self.model, 
            name_export
        )
    
    def loadModel(self, name_model="trained_model.joblib"):

        self.model = load(name_model)

    def makePredictionsWithModel(self, X_matrix=None):

        return self.model.predict(X_matrix)