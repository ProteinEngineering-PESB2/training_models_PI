from sklearn.model_selection import train_test_split

def applySplit(dataset, response, random_state=42, test_size=0.3):

    X_train, X_test, y_train, y_test = train_test_split(
        dataset, 
        response, 
        random_state=random_state, 
        test_size=test_size)
    
    return X_train, X_test, y_train, y_test