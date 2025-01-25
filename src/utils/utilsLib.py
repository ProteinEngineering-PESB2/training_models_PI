from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error

def applySplit(dataset, response, random_state=42, test_size=0.3):

    X_train, X_test, y_train, y_test = train_test_split(
        dataset, 
        response, 
        random_state=random_state, 
        test_size=test_size)
    
    return X_train, X_test, y_train, y_test

def makeScoresForRegression():
    scoring = {
        'MAE': make_scorer(mean_absolute_error),
        'MSE': make_scorer(mean_squared_error),
        'R2': make_scorer(r2_score),
        'RMSE' : make_scorer(root_mean_squared_error)
    }

    return scoring