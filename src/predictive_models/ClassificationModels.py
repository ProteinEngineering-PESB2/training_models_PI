from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, HistGradientBoostingClassifier,
                              BaggingClassifier, GradientBoostingClassifier)
from sklearn.svm import (SVC, NuSVC, LinearSVC)
from sklearn.linear_model import (RidgeClassifier, LogisticRegression, SGDClassifier)
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis)
from sklearn.naive_bayes import (GaussianNB, CategoricalNB, BernoulliNB)
from sklearn.neighbors import (KNeighborsClassifier, RadiusNeighborsClassifier)
from sklearn.gaussian_process import GaussianProcessClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from PredictiveModel import PredictiveModel

class ClassificationModels(PredictiveModel):

    def __init__(
            self, 
            dataset, 
            response, 
            test_size=0.3, 
            random_state=42):

        super().__init__(
            dataset, 
            response, 
            test_size, 
            random_state)
        
    def instanceDecisionTree(self):
        self.model = DecisionTreeClassifier()
    
    def instanceRandomForest(self):
        self.model = RandomForestClassifier()
    
    def instanceAdaBoost(self):
        self.model = AdaBoostClassifier()
    
    def instanceHistGradientBoosting(self):
        self.model = HistGradientBoostingClassifier()
    
    def instanceBagging(self):
        self.model = BaggingClassifier()
    
    def instanceGradientBoosting(self):
        self.model = GradientBoostingClassifier()
    
    def instanceSVC(self):
        self.model = SVC()
    
    def instanceNuSVC(self):
        self.model = NuSVC()
    
    def instanceLinearSVC(self):
        self.model = LinearSVC()
    
    def instanceRidge(self):
        self.model = RidgeClassifier()
    
    def instanceLogisticRegresion(self):
        self.model = LogisticRegression()
    
    def instanceSGD(self):
        self.model = SGDClassifier()
    
    def instanceLDA(self):
        self.model = LinearDiscriminantAnalysis()
    
    def instanceQDA(self):
        self.model = QuadraticDiscriminantAnalysis()
    
    def instanceGaussianNB(self):
        self.model = GaussianNB()
    
    def instanceCategoricalNB(self):
        self.model = CategoricalNB()
    
    def instanceBernoulliNB(self):
        self.model = BernoulliNB()
    
    def instanceNeighbors(self):
        self.model = KNeighborsClassifier()
    
    def instanceRadiusNeighbors(self):
        self.model = RadiusNeighborsClassifier()
    
    def instanceGaussianProcess(self):
        self.model = GaussianProcessClassifier()
    
    def instanceXGBoost(self):
        self.model = XGBClassifier()
    
    def instanceLGBM(self):
        self.model = LGBMClassifier()

    def processModel(
            self,
            averge="weighted",
            normalized_cm="true"):

        self.trainModel()
        self.evalModel(averge=averge, normalized_cm=normalized_cm)