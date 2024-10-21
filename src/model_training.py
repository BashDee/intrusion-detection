import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, chi2
import joblib
import os

# Save model to disk
def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

# Train models
def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train):
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    return model

def train_knn(X_train, y_train):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    return model

# Evaluate models
def evaluate_model(model, X_test, y_test, model_name='Model'):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"{model_name} Accuracy: {acc}")
    print(f"{model_name} Classification Report:\n{report}")
    return acc, report

# Main script
if __name__ == '__main__':
    from data_preprocessing import load_data, preprocess_data

    # Load and preprocess data
    train_data, test_data = load_data('../datasets/NSL_KDD_Train.csv', '../datasets/NSL_KDD_Test.csv')
    X_train, X_test, y_train, y_test, scaler = preprocess_data(train_data, test_data, label_column='label')

    # You do not need to apply pd.get_dummies() here since it's already done in preprocess_data
    # Proceed directly to feature selection

    # Feature selection using SelectKBest
    selector = SelectKBest(score_func=chi2, k=10)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    # Save the selected columns after one-hot encoding and feature selection
    selected_columns = X_train.columns[selector.get_support()]
    joblib.dump(selected_columns, '../models/selected_columns.pkl')

    # Train models
    lr_model = train_logistic_regression(X_train_selected, y_train)
    svm_model = train_svm(X_train_selected, y_train)
    knn_model = train_knn(X_train_selected, y_train)

    # Save models and feature selector
    save_model(lr_model, '../models/logistic_regression.pkl')
    save_model(svm_model, '../models/svm_model.pkl')
    save_model(knn_model, '../models/knn_model.pkl')
    save_model(selector, '../models/feature_selector.pkl')

    # Evaluate models
    evaluate_model(lr_model, X_test_selected, y_test, model_name='Logistic Regression')
    evaluate_model(svm_model, X_test_selected, y_test, model_name='SVM')
    evaluate_model(knn_model, X_test_selected, y_test, model_name='K-Nearest Neighbors')

    print("Models trained and saved successfully.")
