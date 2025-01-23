from sklearn.model_selection import train_test_split
from PerformancesModels import *
from joblib import load, dump

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

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.dataset, 
            self.response,
            test_size=self.test_size,
            random_state=self.random_state)
    
    def trainModel(self):

        self.model.fit(self.X_train, self.X_test)
    
    def evalModel(
            self,
            type_model="class",
            averge=None,
            normalized_cm=None):
        
        predictions_model = self.model.predict(self.X_test)

        if type_model == "class":
            self.performances = calculateClassificationMetrics(
                y_true=self.y_test, 
                y_pred=predictions_model, 
                averge=averge,
                normalized_cm=normalized_cm
            )
        else:
            self.performances = calculateRegressionMetrics(
                y_true=self.y_test, 
                y_pred=predictions_model
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