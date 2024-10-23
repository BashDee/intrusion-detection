import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, MinMaxScaler
import joblib


def load_data(train_path, test_path):
    """Loads the training and testing datasets."""
    col_names = ["duration", "protocol_type", "service", "flag", "src_bytes",
                 "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
                 "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
                 "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
                 "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
                 "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                 "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                 "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                 "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                 "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]
    train_data = pd.read_csv(train_path, header = None, names = col_names)
    test_data = pd.read_csv(test_path, header = None, names = col_names)
    return train_data, test_data


def preprocess_data(train_data, test_data, label_column='label'):
    categorical_columns = ['protocol_type', 'service', 'flag']
    
    # One-hot encode categorical features
    train_data_categorical = pd.get_dummies(train_data[categorical_columns])
    test_data_categorical = pd.get_dummies(test_data[categorical_columns])
    print("Train Data: ", train_data_categorical)
    print(test_data_categorical)

    # Align the test set with the training set (fill missing columns with 0)
    test_data_categorical = test_data_categorical.reindex(columns=train_data_categorical.columns, fill_value=0)
    print(test_data_categorical)
    
    # Drop the original categorical columns and join encoded columns
    train_data = train_data.drop(categorical_columns, axis=1).join(train_data_categorical)
    test_data = test_data.drop(categorical_columns, axis=1).join(test_data_categorical)
    print(train_data)
    print(test_data)

    # # Scale numerical features
    # scaler = MinMaxScaler()
    # X_train = scaler.fit_transform(train_data.drop(label_column, axis=1))
    # X_test = scaler.transform(test_data.drop(label_column, axis=1))
    
    # # Save the scaler for future use
    # joblib.dump(scaler, '../models/scaler.pkl')
    
    # Return features and labels
    X_train = train_data.drop(label_column, axis=1)
    y_train = train_data[label_column]
    X_test = test_data.drop(label_column, axis=1)
    y_test = test_data[label_column]
    
    return X_train, X_test, y_train, y_test
