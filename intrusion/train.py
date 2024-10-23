import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load your data (adjust paths as needed)
train_data = pd.read_csv('NSL_KDD_Train.csv')
test_data = pd.read_csv('NSL_KDD_Test.csv')

# Preprocessing function (make sure to adapt based on your actual data structure)
def preprocess_data(data, label_column):
    X = data.drop(columns=[label_column])
    y = data[label_column]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

X_train, y_train, scaler_train = preprocess_data(train_data, 'target')
X_test, y_test, scaler_test = preprocess_data(test_data, 'target')

# Train the Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'logistic_regression_model.pkl')

# Evaluate the model
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f"Accuracy: {acc}")
print(f"Classification Report:\n{report}")
