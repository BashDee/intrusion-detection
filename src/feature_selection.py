from sklearn.feature_selection import SelectKBest, chi2
import joblib


def select_features(X_train, y_train, X_test, k=10):
    """Selects the top k features based on the Chi-Squared test."""
    selector = SelectKBest(score_func = chi2, k = k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    return X_train_selected, X_test_selected, selector


def save_selector(selector, path='../models/feature_selector.pkl'):
    """Saves the feature selector to disk."""
    joblib.dump(selector, path)


def load_selector(path='../models/feature_selector.pkl'):
    """Loads the feature selector from disk."""
    return joblib.load(path)


# Example usage
if __name__ == '__main__':
    from data_preprocessing import load_data, preprocess_data

    # Load and preprocess data
    train_data, test_data = load_data('../datasets/NSL_KDD_Train.csv', '../datasets/NSL_KDD_Test.csv')
    X_train, X_test, y_train, y_test, scaler = preprocess_data(train_data, test_data, label_column='label')

    # Feature selection
    X_train_selected, X_test_selected, selector = select_features(X_train, y_train, X_test, k=10)

    # Save the selector
    save_selector(selector)

    print("Feature selection complete and selector saved.")
