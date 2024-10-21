from data_preprocessing import load_data, preprocess_data
from feature_selection import select_features
from model_training import train_logistic_regression, evaluate_model
from unsupervised_models import train_kmeans, evaluate_kmeans
from alerting import alert_system
import joblib

if __name__ == '__main__':
    # Step 1: Load and preprocess data
    data = load_data('NSL_KDD.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)

    # Step 2: Feature Selection
    X_train_selected, selector = select_features(X_train, y_train)

    # Step 3: Train Logistic Regression and evaluate
    lr_model = train_logistic_regression(X_train_selected, y_train)
    evaluate_model(lr_model, X_test, y_test)

    # Step 4: Train and evaluate K-Means
    kmeans_model = train_kmeans(X_train_selected)
    kmeans_labels = evaluate_kmeans(kmeans_model, X_test)

    # Step 5: Alerting System for attacks
    for label in kmeans_labels:
        alert_system(label)
