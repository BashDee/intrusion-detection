import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np



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
    """
    Preprocess the datasets: handle missing values, encode categorical variables,
    normalize numerical features, and split into features and labels.
    """
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    categorical_columns = ['protocol_type', 'service', 'flag']

    train_data_categorical_values = train_data[categorical_columns]
    test_data_categorical_values = test_data[categorical_columns]

    train_data_categorical_values.head()
    test_data_categorical_values.head()

    # protocol type
    unique_protocol = sorted(train_data.protocol_type.unique())
    string1 = 'Protocol_type_'
    unique_protocol2 = [string1 + x for x in unique_protocol]
    print(unique_protocol2)

    unique_protocol_test = sorted(test_data.protocol_type.unique())
    unique_protocol2_test = [string1 + x for x in unique_protocol_test]
    print(unique_protocol2_test)

    # service
    unique_service = sorted(train_data.service.unique())
    string2 = 'service_'
    unique_service2 = [string2 + x for x in unique_service]
    print(unique_service2)

    unique_service_test = sorted(test_data.service.unique())
    unique_service2_test = [string2 + x for x in unique_service_test]
    print(unique_service2_test)

    # flag
    unique_flag = sorted(train_data.flag.unique())
    string3 = 'flag_'
    unique_flag2 = [string3 + x for x in unique_flag]
    print(unique_flag2)

    unique_flag_test = sorted(test_data.flag.unique())
    unique_flag2_test = [string3 + x for x in unique_flag_test]
    print(unique_flag2_test)

    # put together
    dumcols = unique_protocol2 + unique_service2 + unique_flag2
    testdumcols = unique_protocol2_test + unique_service2_test + unique_flag2_test

    train_data_categorical_values_enc = train_data_categorical_values.apply(LabelEncoder().fit_transform)
    test_data_categorical_values_enc = test_data_categorical_values.apply(LabelEncoder().fit_transform)

    enc = OneHotEncoder(categories='auto')
    train_data_categorical_values_encenc = enc.fit_transform(train_data_categorical_values_enc)
    train_data_cat_data = pd.DataFrame(train_data_categorical_values_encenc.toarray(), columns=dumcols)

    test_data_categorical_values_encenc = enc.fit_transform(test_data_categorical_values_enc)
    test_data_cat_data = pd.DataFrame(test_data_categorical_values_encenc.toarray(), columns=testdumcols)

    train_data_cat_data.head()
    test_data_cat_data.head()

    trainservice = train_data['service'].tolist()
    testservice = test_data['service'].tolist()
    difference = list(set(trainservice) - set(testservice))
    string = 'service_'
    difference = [string + x for x in difference]


    for col in difference:
        test_data_cat_data[col] = 0

    print(train_data_cat_data.shape)
    print(test_data_cat_data.shape)

    new_train_data = train_data.join(train_data_cat_data).drop(categorical_columns, axis=1)
    new_test_data = test_data.join(test_data_cat_data).drop(categorical_columns, axis=1)


    print(new_train_data.shape)
    print(new_test_data.shape)

    # Align the test data to the train data columns
    new_test_data = new_test_data.reindex(columns=new_train_data.columns, fill_value=0)

    # Split into features and target
    X_train = new_train_data.drop(label_column, axis=1)
    y_train = new_train_data[label_column]
    X_test = new_test_data.drop(label_column, axis=1)
    y_test = new_test_data[label_column]


    # Normalize numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert NumPy arrays back to DataFrames after scaling
    X_train = pd.DataFrame(X_train, columns=new_train_data.drop(label_column, axis=1).columns)
    X_test = pd.DataFrame(X_test, columns=new_train_data.drop(label_column, axis=1).columns)

    
    # Ensure all features are non-negative
    X_train = np.abs(X_train)
    X_test = np.abs(X_test)

    # Save the scaler for future use
    joblib.dump(scaler, '../models/scaler.pkl')

    return X_train, X_test, y_train, y_test, scaler

# Example usage
if __name__ == '__main__':
    train_data, test_data = load_data('../datasets/NSL_KDD_Train.csv', '../datasets/NSL_KDD_Test.csv')
    X_train, X_test, y_train, y_test, scaler = preprocess_data(train_data, test_data, label_column='label')
    print("Preprocessing complete.")
