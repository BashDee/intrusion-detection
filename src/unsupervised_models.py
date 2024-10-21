from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def train_kmeans(X_train, n_clusters=2):
    """Train K-Means clustering and return the model"""
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(X_train)
    return model

def evaluate_kmeans(model, X_test):
    """Evaluate K-Means clustering using silhouette score"""
    labels = model.predict(X_test)
    score = silhouette_score(X_test, labels)
    print(f"Silhouette Score: {score}")
    return labels

# Usage Example
if __name__ == '__main__':
    from data_preprocessing import load_data, preprocess_data
    from feature_selection import select_features

    # Load and preprocess data
    data = load_data('../datasets/NSL_KDD.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    X_train_selected, selector = select_features(X_train, y_train)

    # Train and evaluate K-Means clustering
    kmeans_model = train_kmeans(X_train_selected)
    evaluate_kmeans(kmeans_model, X_test)
