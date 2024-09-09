from sklearn.cluster import KMeans

def kmeans_clustering(X, n_clusters=2):
    """
    Realiza el clustering en los datos utilizando el algoritmo K-Means.
        
    devuelve lo que es:
    - clusters: que son etiquetas de cluster asignadas a cada punto en X.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    return clusters