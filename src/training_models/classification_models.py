from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from training_models.predictive_models import PredictiveModel


class ClassificationModels(PredictiveModel):
    def __init__(self, X_train=None, X_val=None, y_train=None, y_val=None):
        super().__init__(X_train, X_val, y_train, y_val)

    def instance_decision_tree(
        self,
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0,
        class_weight=None,
        ccp_alpha=0,
    ):
        self.model = DecisionTreeClassifier(
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
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
        )

    def instance_random_forest(
        self,
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0,
        max_samples=None,
    ):
        self.model = RandomForestClassifier(
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
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
        )

    def instance_ada_boost(
        self, estimator=None, n_estimators=50, learning_rate=1, algorithm="SAMME", random_state=None
    ):
        self.model = AdaBoostClassifier(
            estimator=estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm=algorithm,
            random_state=random_state,
        )

    def instance_hist_gradient_boosting(
        self,
        loss="log_loss",
        learning_rate=0.1,
        max_iter=100,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=20,
        l2_regularization=0,
        max_features=1.0,
        max_bins=255,
        categorical_features=None,
        monotonic_cst=None,
        interaction_cst=None,
        warm_start=False,
        early_stopping="auto",
        scoring="loss",
        validation_fraction=0.1,
        n_iter_no_change=10,
        tol=1e-7,
        verbose=0,
        random_state=None,
        class_weight=None,
    ):
        self.model = HistGradientBoostingClassifier(
            loss=loss,
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
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
            random_state=random_state,
            class_weight=class_weight,
        )

    def instance_bagging(
        self,
        estimator=None,
        n_estimators=10,
        max_samples=1,
        max_features=1,
        bootstrap=True,
        bootstrap_features=False,
        oob_score=False,
        warm_start=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
    ):
        self.model = BaggingClassifier(
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
            verbose=verbose,
        )

    def instance_gradient_boosting(
        self,
        loss="log_loss",
        learning_rate=0.1,
        n_estimators=100,
        subsample=1,
        criterion="friedman_mse",
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0,
        max_depth=3,
        min_impurity_decrease=0,
        init=None,
        random_state=None,
        max_features=None,
        verbose=0,
        max_leaf_nodes=None,
        warm_start=False,
        validation_fraction=0.1,
        n_iter_no_change=None,
        tol=0.0001,
        ccp_alpha=0,
    ):
        self.model = GradientBoostingClassifier(
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
            verbose=verbose,
            max_leaf_nodes=max_leaf_nodes,
            warm_start=warm_start,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            ccp_alpha=ccp_alpha,
        )

    def instance_svc(
        self,
        C=1,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0,
        shrinking=True,
        probability=False,
        tol=0.001,
        cache_size=200,
        class_weight=None,
        verbose=False,
        decision_function_shape="ovr",
        break_ties=False,
        random_state=None,
    ):
        self.model = SVC(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            shrinking=shrinking,
            probability=probability,
            tol=tol,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=verbose,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=random_state,
        )

    def instance_nu_svc(
        self,
        nu=0.5,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0,
        shrinking=True,
        probability=False,
        tol=0.001,
        cache_size=200,
        class_weight=None,
        verbose=False,
        decision_function_shape="ovr",
        break_ties=False,
        random_state=None,
    ):
        self.model = NuSVC(
            nu=nu,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            shrinking=shrinking,
            probability=probability,
            tol=tol,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=verbose,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=random_state,
        )

    def instance_linear_svc(
        self,
        penalty="l2",
        loss="squared_hinge",
        dual=True,
        tol=0.0001,
        C=1,
        multi_class="ovr",
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        verbose=0,
        random_state=None,
        max_iter=1000,
    ):
        self.model = LinearSVC(
            penalty=penalty,
            loss=loss,
            dual=dual,
            tol=tol,
            C=C,
            multi_class=multi_class,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            class_weight=class_weight,
            verbose=verbose,
            random_state=random_state,
            max_iter=max_iter,
        )

    def instance_ridge(
        self,
        alpha=1,
        fit_intercept=True,
        copy_X=True,
        max_iter=None,
        tol=0.0001,
        class_weight=None,
        solver="auto",
        positive=False,
        random_state=None,
    ):
        self.model = RidgeClassifier(
            alpha=alpha,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            max_iter=max_iter,
            tol=tol,
            class_weight=class_weight,
            solver=solver,
            positive=positive,
            random_state=random_state,
        )

    def instance_logistic_regresion(
        self,
        penalty="l2",
        dual=False,
        tol=0.0001,
        C=1,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        solver="lbfgs",
        max_iter=100,
        multi_class="auto",
        verbose=0,
        warm_start=False,
        n_jobs=None,
        l1_ratio=None,
    ):
        self.model = LogisticRegression(
            penalty=penalty,
            dual=dual,
            tol=tol,
            C=C,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            class_weight=class_weight,
            random_state=random_state,
            solver=solver,
            max_iter=max_iter,
            multi_class=multi_class,
            verbose=verbose,
            warm_start=warm_start,
            n_jobs=n_jobs,
            l1_ratio=l1_ratio,
        )

    def instance_sgd(
        self,
        loss="hinge",
        penalty="l2",
        alpha=0.0001,
        l1_ratio=0.15,
        fit_intercept=True,
        max_iter=1000,
        tol=0.001,
        shuffle=True,
        verbose=0,
        n_jobs=None,
        random_state=None,
        learning_rate="optimal",
        eta0=0,
        power_t=0.5,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=5,
        class_weight=None,
        warm_start=False,
        average=False,
    ):
        self.model = SGDClassifier(
            loss=loss,
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            shuffle=shuffle,
            verbose=verbose,
            n_jobs=n_jobs,
            random_state=random_state,
            learning_rate=learning_rate,
            eta0=eta0,
            power_t=power_t,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            class_weight=class_weight,
            warm_start=warm_start,
            average=average,
        )

    def instance_lda(
        self,
        solver="svd",
        shrinkage=None,
        priors=None,
        n_components=None,
        store_covariance=False,
        tol=0.0001,
        covariance_estimator=None,
    ):
        self.model = LinearDiscriminantAnalysis(
            solver=solver,
            shrinkage=shrinkage,
            priors=priors,
            n_components=n_components,
            store_covariance=store_covariance,
            tol=tol,
            covariance_estimator=covariance_estimator,
        )

    def instance_qda(self, priors=None, reg_param=0, store_covariance=False, tol=0.0001):
        self.model = QuadraticDiscriminantAnalysis(
            priors=priors, reg_param=reg_param, store_covariance=store_covariance, tol=tol
        )

    def instance_gaussian_nb(self, priors=None, var_smoothing=1e-9):
        self.model = GaussianNB(priors=priors, var_smoothing=var_smoothing)

    def instance_categorical_nb(
        self, alpha=1, force_alpha="warn", fit_prior=True, class_prior=None, min_categories=None
    ):
        self.model = CategoricalNB(
            alpha=alpha,
            force_alpha=force_alpha,
            fit_prior=fit_prior,
            class_prior=class_prior,
            min_categories=min_categories,
        )

    def instance_bernoulli_nb(
        self, alpha=1, force_alpha="warn", binarize=0, fit_prior=True, class_prior=None
    ):
        self.model = BernoulliNB(
            alpha=alpha,
            force_alpha=force_alpha,
            binarize=binarize,
            fit_prior=fit_prior,
            class_prior=class_prior,
        )

    def instance_neighbors(
        self,
        n_neighbors=5,
        weights="uniform",
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
        metric_params=None,
        n_jobs=None,
    ):
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )

    def instance_radius_neighbors(
        self,
        radius=1,
        weights="uniform",
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
        outlier_label=None,
        metric_params=None,
        n_jobs=None,
    ):
        self.model = RadiusNeighborsClassifier(
            radius=radius,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric,
            outlier_label=outlier_label,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )

    def instance_gaussian_process(
        self,
        kernel=None,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=0,
        max_iter_predict=100,
        warm_start=False,
        copy_X_train=True,
        random_state=None,
        multi_class="one_vs_rest",
        n_jobs=None,
    ):
        self.model = GaussianProcessClassifier(
            kernel=kernel,
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts_optimizer,
            max_iter_predict=max_iter_predict,
            warm_start=warm_start,
            copy_X_train=copy_X_train,
            random_state=random_state,
            multi_class=multi_class,
            n_jobs=n_jobs,
        )

    def instance_xg_boost(self):
        self.model = XGBClassifier()

    def instance_lgbm(
        self,
        boosting_type="gbdt",
        num_leaves=31,
        max_depth=-1,
        learning_rate=0.1,
        n_estimators=100,
        subsample_for_bin=200000,
        objective=None,
        class_weight=None,
        min_split_gain=0,
        min_child_weight=0.001,
        min_child_samples=20,
        subsample=1,
        subsample_freq=0,
        colsample_bytree=1,
        reg_alpha=0,
        reg_lambda=0,
        random_state=None,
        n_jobs=None,
        importance_type="split",
    ):
        self.model = LGBMClassifier(
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
            importance_type=importance_type,
        )

    def process_model(
        self,
        averge="weighted",
        normalized_cm="true",
        k=10,
        kfold=False,
        stratified=False,
        scores=["f1_weighted", "recall_weighted", "precision_weighted", "accuracy"],
    ):
        if not kfold and not stratified:
            self.train_model()
            self.performances = {
                "validation_metrics": self.eval_model(
                    y_true=self.y_val,
                    y_pred=self.make_predictions_with_model(self.X_val),
                    averge=averge,
                    normalized_cm=normalized_cm,
                )
            }

        else:
            if kfold:
                self.performances = {
                    "training_metrics": self.train_model_with_kfold(
                        scores=scores, k=k, preffix="test_"
                    )
                }
            else:
                self.performances = {
                    "training_metrics": self.train_model_with_kfold(
                        scores=scores, k=k, stratified=True, preffix="test_"
                    )
                }

            self.train_model()
            self.performances.update(
                {
                    "validation_metrics": self.eval_model(
                        y_true=self.y_val,
                        y_pred=self.make_predictions_with_model(self.X_val),
                        averge=averge,
                        normalized_cm=normalized_cm,
                    )
                }
            )
