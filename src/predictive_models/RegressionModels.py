from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor, HistGradientBoostingRegressor,
                              BaggingRegressor, GradientBoostingRegressor)
from sklearn.svm import (SVR, NuSVR, LinearSVR)
from sklearn.linear_model import (Ridge, SGDRegressor, RANSACRegressor, ARDRegression, BayesianRidge,
                                  ElasticNet, GammaRegressor, HuberRegressor, Lars, Lasso, LassoLars, LinearRegression,
                                  OrthogonalMatchingPursuit, PoissonRegressor, QuantileRegressor, TheilSenRegressor, TweedieRegressor)
from sklearn.neighbors import (KNeighborsRegressor, RadiusNeighborsRegressor)
from sklearn.gaussian_process import GaussianProcessRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from predictive_models.PredictiveModel import PredictiveModel

class RegressionModels(PredictiveModel):

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
    
    def instanceGaussianProcess(
            self,
            kernel= None,
            alpha= 1e-10,
            optimizer= "fmin_l_bfgs_b",
            n_restarts_optimizer= 0,
            normalize_y= False,
            copy_X_train= True,
            random_state= None):
        
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=normalize_y,
            copy_X_train=copy_X_train,
            random_state=random_state)
    
    def instanceXGB(self):
        self.model = XGBRegressor()
    
    def instanceLGBM(
            self,
            boosting_type= "gbdt",
            num_leaves= 31,
            max_depth= -1,
            learning_rate= 0.1,
            n_estimators= 100,
            subsample_for_bin= 200000,
            objective= None,
            class_weight= None,
            min_split_gain= 0,
            min_child_weight= 0.001,
            min_child_samples= 20,
            subsample= 1,
            subsample_freq= 0,
            colsample_bytree= 1,
            reg_alpha= 0,
            reg_lambda= 0,
            random_state= None,
            n_jobs= None,
            importance_type= "split"):
        
        self.model = LGBMRegressor(
            boosting_type=boosting_type,
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample_for_bin=subsample_for_bin,
            objective=objective,
            class_weight=class_weight,
            min_split_gain=min_split_gain,
            min_child_weight=min_child_weight,
            min_child_samples=min_child_samples,
            subsample=subsample,
            subsample_freq=subsample_freq,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
            importance_type=importance_type)

    def instanceDecisionTree(
            self,
            criterion= "squared_error",
            splitter= "best",
            max_depth= None,
            min_samples_split= 2,
            min_samples_leaf= 1,
            min_weight_fraction_leaf= 0,
            max_features= None,
            random_state= None,
            max_leaf_nodes= None,
            min_impurity_decrease= 0,
            ccp_alpha= 0):
        
        self.model = DecisionTreeRegressor(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            random_state=random_state,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha)
    
    def instanceRandomForest(
            self,
            n_estimators= 100,
            criterion= "squared_error",
            max_depth= None,
            min_samples_split= 2,
            min_samples_leaf= 1,
            min_weight_fraction_leaf= 0,
            max_features= 1,
            max_leaf_nodes= None,
            min_impurity_decrease= 0,
            bootstrap= True,
            oob_score= False,
            n_jobs= None,
            random_state= None,
            verbose= 0,
            warm_start= False,
            ccp_alpha= 0,
            max_samples= None):
        
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples)
    
    def instanceAdaBoost(
            self,
            estimator= None,
            n_estimators= 50,
            learning_rate= 1,
            loss= "linear",
            random_state= None):
        
        self.model = AdaBoostRegressor(
            estimator=estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            loss=loss,
            random_state=random_state)
    
    def instanceHistGradient(
            self,
            loss= "squared_error",
            quantile= None,
            learning_rate= 0.1,
            max_iter= 100,
            max_leaf_nodes= 31,
            max_depth= None,
            min_samples_leaf= 20,
            l2_regularization= 0,
            max_features= 1,
            max_bins= 255,
            categorical_features= None,
            monotonic_cst= None,
            interaction_cst= None,
            warm_start= False,
            early_stopping= "auto",
            scoring= "loss",
            validation_fraction= 0.1,
            n_iter_no_change= 10,
            tol= 1e-7,
            verbose= 0,
            random_state= None):
        
        self.model = HistGradientBoostingRegressor(
            loss=loss,
            quantile=quantile,
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            l2_regularization=l2_regularization,
            max_features=max_features,
            max_bins=max_bins,
            categorical_features=categorical_features,
            monotonic_cst=monotonic_cst,
            interaction_cst=interaction_cst,
            warm_start=warm_start,
            early_stopping=early_stopping,
            scoring=scoring,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            verbose=verbose,
            random_state=random_state)
    
    def instanceBagging(
            self,
            estimator= None,
            n_estimators= 10,
            max_samples= 1,
            max_features= 1,
            bootstrap= True,
            bootstrap_features= False,
            oob_score= False,
            warm_start= False,
            n_jobs= None,
            random_state= None,
            verbose= 0):
        
        self.model = BaggingRegressor(
            estimator=estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)
    
    def instanceGradientBoosting(
            self,
            loss= "squared_error",
            learning_rate= 0.1,
            n_estimators= 100,
            subsample= 1,
            criterion= "friedman_mse",
            min_samples_split= 2,
            min_samples_leaf= 1,
            min_weight_fraction_leaf= 0,
            max_depth = 3,
            min_impurity_decrease= 0,
            init = None,
            random_state= None,
            max_features= None,
            alpha= 0.9,
            verbose= 0,
            max_leaf_nodes = None,
            warm_start= False,
            validation_fraction= 0.1,
            n_iter_no_change = None,
            tol= 0.0001,
            ccp_alpha= 0):
        
        self.model = GradientBoostingRegressor(
            loss=loss,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth,
            min_impurity_decrease=min_impurity_decrease,
            init=init,
            random_state=random_state,
            max_features=max_features,
            alpha=alpha,
            verbose=verbose,
            max_leaf_nodes=max_leaf_nodes,
            warm_start=warm_start,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            ccp_alpha=ccp_alpha)
    
    def instanceSVR(
            self,
            kernel= "rbf",
            degree= 3,
            gamma= "scale",
            tol= 0.001,
            C= 1,
            epsilon= 0.1,
            shrinking= True,
            cache_size= 200,
            verbose= False):
        
        self.model = SVR(
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            tol=tol,
            C=C,
            epsilon=epsilon,
            shrinking=shrinking,
            cache_size=cache_size,
            verbose=verbose) 
    
    def instanceNuSVR(
            self,
            nu=0.5,
            C=1,
            kernel="rbf",
            degree= 3,
            gamma= "scale",
            coef0= 0,
            shrinking= True,
            tol= 0.001,
            cache_size= 200,
            verbose= False):
        
        self.model = NuSVR(
            nu=nu,
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            shrinking=shrinking,
            tol=tol,
            cache_size=cache_size,
            verbose=verbose)
    
    def instanceLinearSVR(
            self,
            epsilon= 0,
            tol= 0.0001,
            C= 1,
            loss= "epsilon_insensitive",
            fit_intercept= True,
            intercept_scaling= 1,
            dual= True,
            verbose= 0,
            random_state= None,
            max_iter= 1000):
        
        self.model = LinearSVR(
            epsilon=epsilon,
            tol=tol,
            C=C,
            loss=loss,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            dual=dual,
            verbose=verbose,
            random_state=random_state,
            max_iter=max_iter)
    
    def instanceRidge(
            self,
            alpha= 1,
            fit_intercept= True,
            copy_X= True,
            max_iter= None,
            tol= 0.0001,
            solver= "auto",
            positive= False,
            random_state= None):
        
        self.model = Ridge(
            alpha=alpha,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            max_iter=max_iter,
            tol=tol,
            solver=solver,
            positive=positive,
            random_state=random_state) 
    
    def instanceSGD(
            self,
            loss= "squared_error",
            penalty= "l2",
            alpha= 0.0001,
            l1_ratio= 0.15,
            fit_intercept= True,
            max_iter= 1000,
            tol= 0.001,
            shuffle= True,
            verbose= 0,
            random_state= None,
            learning_rate= "invscaling",
            eta0= 0.01,
            power_t= 0.25,
            early_stopping= False,
            validation_fraction= 0.1,
            n_iter_no_change= 5,
            warm_start= False,
            average= False):
        
        self.model = SGDRegressor(
            loss=loss,
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            shuffle=shuffle,
            verbose=verbose,
            random_state=random_state,
            learning_rate=learning_rate,
            eta0=eta0,
            power_t=power_t,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            warm_start=warm_start,
            average=average)
    
    def instanceRANSAC(
            self,
            estimator= None,
            min_samples= None,
            residual_threshold= None,
            is_data_valid= None,
            is_model_valid= None,
            max_trials= 100,
            stop_probability= 0.99,
            loss= "absolute_error",
            random_state= None):
        
        self.model = RANSACRegressor(
            estimator=estimator,
            min_samples=min_samples,
            residual_threshold=residual_threshold,
            is_data_valid=is_data_valid,
            is_model_valid=is_model_valid,
            max_trials=max_trials,
            stop_probability=stop_probability,
            loss=loss,
            random_state=random_state)
    
    def instanceARDR(
            self,
            n_iter= 300,
            tol= 0.001,
            alpha_1= 0.000001,
            alpha_2= 0.000001,
            lambda_1= 0.000001,
            lambda_2= 0.000001,
            compute_score= False,
            threshold_lambda= 10000,
            fit_intercept= True,
            copy_X= True,
            verbose= False):
        
        self.model = ARDRegression(
            n_iter=n_iter,
            tol=tol,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            compute_score=compute_score,
            threshold_lambda=threshold_lambda,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            verbose=verbose)
    
    def instanceBayesian(
            self,
            n_iter= 300,
            tol= 0.001,
            alpha_1= 0.000001,
            alpha_2= 0.000001,
            lambda_1= 0.000001,
            lambda_2= 0.000001,
            alpha_init= None,
            lambda_init= None,
            compute_score= False,
            fit_intercept= True,
            copy_X= True,
            verbose= False):
        
        self.model = BayesianRidge(
            n_iter=n_iter,
            tol=tol,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            alpha_init=alpha_init,
            lambda_init=lambda_init,
            compute_score=compute_score,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            verbose=verbose)
    
    def instanceElastic(
            self,
            alpha= 1,
            l1_ratio= 0.5,
            fit_intercept= True,
            precompute= False,
            max_iter= 1000,
            copy_X= True,
            tol= 0.0001,
            warm_start= False,
            positive= False,
            random_state= None,
            selection= "cyclic"):
        
        self.model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            precompute=precompute,
            max_iter=max_iter,
            copy_X=copy_X,
            tol=tol,
            warm_start=warm_start,
            positive=positive,
            random_state=random_state,
            selection=selection)
    
    def instanceGamma(
            self,
            alpha= 1,
            fit_intercept= True,
            solver= "lbfgs",
            max_iter= 100,
            tol= 0.0001,
            warm_start= False,
            verbose= 0):
        
        self.model = GammaRegressor(
            alpha=alpha,
            fit_intercept=fit_intercept,
            solver=solver,
            max_iter=max_iter,
            tol=tol,
            warm_start=warm_start,
            verbose=verbose)
    
    def instanceHuber(
            self,
            epsilon= 1.35,
            max_iter= 100,
            alpha= 0.0001,
            warm_start= False,
            fit_intercept= True,
            tol= 0.00001):
        
        self.model = HuberRegressor(
            epsilon=epsilon,
            max_iter=max_iter,
            alpha=alpha,
            warm_start=warm_start,
            fit_intercept=fit_intercept,
            tol=tol)
    
    def instanceLars(
            self,
            fit_intercept= True,
            verbose= False,
            precompute= "auto",
            n_nonzero_coefs= 500,
            copy_X= True,
            fit_path= True,
            jitter= None,
            random_state= None):
        
        self.model = Lars(
            fit_intercept=fit_intercept,
            verbose=verbose,
            precompute=precompute,
            n_nonzero_coefs=n_nonzero_coefs,
            copy_X=copy_X,
            fit_path=fit_path,
            jitter=jitter,
            random_state=random_state)
    
    def instanceLasso(
            self,
            alpha= 1,
            fit_intercept= True,
            precompute= False,
            copy_X= True,
            max_iter= 1000,
            tol= 0.0001,
            warm_start= False,
            positive= False,
            random_state= None,
            selection= "cyclic"):
        
        self.model = Lasso(
            alpha=alpha,
            fit_intercept=fit_intercept,
            precompute=precompute,
            copy_X=copy_X,
            max_iter=max_iter,
            tol=tol,
            warm_start=warm_start,
            positive=positive,
            random_state=random_state,
            selection=selection)
    
    def instanceLassoLars(
            self,
            alpha= 1,
            fit_intercept= True,
            verbose= False,
            normalize= "deprecated",
            precompute= "auto",
            max_iter= 500,
            copy_X= True,
            fit_path= True,
            positive= False,
            jitter= None,
            random_state= None):
        
        self.model = LassoLars(
            alpha=alpha,
            fit_intercept=fit_intercept,
            verbose=verbose,
            normalize=normalize,
            precompute=precompute,
            max_iter=max_iter,
            copy_X=copy_X,
            fit_path=fit_path,
            positive=positive,
            jitter=jitter,
            random_state=random_state)
    
    def instanceLinearRegression(
            self,
            fit_intercept= True,
            copy_X= True,
            n_jobs= None,
            positive= False):
        
        self.model = LinearRegression(
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            n_jobs=n_jobs,
            positive=positive)
    
    def instanceOrthogonal(
            self,
            n_nonzero_coefs= None,
            tol= None,
            fit_intercept= True,
            normalize= "deprecated",
            precompute= "auto"):
        
        self.model = OrthogonalMatchingPursuit(
            n_nonzero_coefs=n_nonzero_coefs,
            tol=tol,
            fit_intercept=fit_intercept,
            normalize=normalize,
            precompute=precompute)
    
    def instancePoisson(
            self,
            alpha= 1,
            fit_intercept= True,
            solver= "lbfgs",
            max_iter= 100,
            tol= 0.0001,
            warm_start= False,
            verbose= 0):
        
        self.model = PoissonRegressor(
            alpha=alpha,
            fit_intercept=fit_intercept,
            solver=solver,
            max_iter=max_iter,
            tol=tol,
            warm_start=warm_start,
            verbose=verbose)
    
    def instanceQuantile(
            self,
            quantile= 0.5,
            alpha= 1,
            fit_intercept= True,
            solver= "warn",
            solver_options= None):
        
        self.model = QuantileRegressor(
            quantile=quantile,
            alpha=alpha,
            fit_intercept=fit_intercept,
            solver=solver,
            solver_options=solver_options)
    
    def instanceTheil(
            self,
            fit_intercept= True,
            copy_X= True,
            max_subpopulation= 10000,
            n_subsamples= None,
            max_iter= 300,
            tol= 0.001,
            random_state= None,
            n_jobs= None,
            verbose= False):
        
        self.model = TheilSenRegressor(
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            max_subpopulation=max_subpopulation,
            n_subsamples=n_subsamples,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose)
    
    def instanceTweedie(
            self,
            power= 0,
            alpha= 1,
            fit_intercept= True,
            link= "auto",
            solver= "lbfgs",
            max_iter= 100,
            tol= 0.0001,
            warm_start= False,
            verbose= 0):
        
        self.model = TweedieRegressor(
            power=power,
            alpha=alpha,
            fit_intercept=fit_intercept,
            link=link,
            solver=solver,
            max_iter=max_iter,
            tol=tol,
            warm_start=warm_start,
            verbose=verbose)

    def instanceKNeighbors(
            self,
            n_neighbors= 5,
            weights= "uniform",
            algorithm= "auto",
            leaf_size= 30,
            p= 2,
            metric= "minkowski",
            metric_params= None,
            n_jobs= None):
        
        self.model = KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric,
            metric_params=metric_params,
            n_jobs=n_jobs)
    
    def instanceRadiusNeighbors(
            self,
            radius= 1,
            weights= "uniform",
            algorithm= "auto",
            leaf_size= 30,
            p= 2,
            metric= "minkowski",
            metric_params= None,
            n_jobs=None):
        
        self.model = RadiusNeighborsRegressor(
            radius=radius,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric,
            metric_params=metric_params,
            n_jobs=n_jobs)
    
    def processModel(
            self):

        self.splitDataset()
        self.trainModel()
        self.performances = self.evalModel(
            type_model="regx",
            y_true=self.y_test,
            y_pred=self.makePredictionsWithModel(self.X_test),
)