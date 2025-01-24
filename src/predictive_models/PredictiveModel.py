from predictive_models.PerformancesModels import *
from utils.utilsLib import *

from joblib import load, dump
from sklearn.model_selection import cross_validate, StratifiedKFold

class PredictiveModel(object):

    def __init__(
            self, 
            dataset, 
            response,
            test_size=0.3,
            random_state=42):
        
        self.dataset = dataset
        self.response = response
        self.test_size = test_size
        self.random_state = random_state

        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.performances = None

    def splitDataset(self):

        self.X_train, self.X_test, self.y_train, self.y_test = applySplit(
            self.dataset, 
            self.response, 
            random_state=self.random_state, 
            test_size=self.test_size)
        
    def trainModel(self):

        self.model.fit(self.X_train, self.y_train)
    
    def trainModelWithKFold(self, k=10):
        pass

    def trainModelWithStratified(self, k=10):
        pass

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