from utils.data_preparation import prepare_data, split_data
from models.kmeans_model import kmeans_clustering
from models.decision_tree import train_decision_tree, predict_decision_tree
from models.evaluation import evaluate_model
from sklearn.cluster import KMeans
import numpy as np

def main():
    """
    Función principal que prepara los datos, entrena un modelo K-Means para clustering,
    y entrena un arbol de decision para predecir la benignidad o malignidad de tumores.
    Luego evalúa el modelo y permite predicciones interactivas.
    """
    # Preparacion de los datos
    X, y, scaler = prepare_data('data/data.csv')
    X_train, X_validation, X_test, y_train, y_validation, y_test = split_data(X, y)
    
    # Aplicar K-Means y agregar los clusters como una nueva caracteristica
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters_train = kmeans.fit_predict(X_train)
    clusters_validation = kmeans.predict(X_validation)
    clusters_test = kmeans.predict(X_test)
    
    X_train_with_clusters = np.hstack((X_train, clusters_train.reshape(-1, 1)))
    X_validation_with_clusters = np.hstack((X_validation, clusters_validation.reshape(-1, 1)))
    X_test_with_clusters = np.hstack((X_test, clusters_test.reshape(-1, 1)))
    
    # Aqui ya se entrena al arbol de decision
    model = train_decision_tree(X_train_with_clusters, y_train)

    print(f"El Árbol de Decisión tiene una profundidad de: {model.get_depth()}")
    print(f"El Árbol de Decisión tiene {model.get_n_leaves()} hojas.")
    print(f"El Árbol de Decisión tiene un total de {model.tree_.node_count} nodos.")
    
    # Evaluar el modelo en el conjunto de validacion
    print("Evaluación en el conjunto de validación:")
    y_pred_validation = predict_decision_tree(model, X_validation_with_clusters)
    evaluate_model(y_validation, y_pred_validation)
    
    # Aqui se evalua el modelo de conjunto de prueba
    print("\nEvaluación en el conjunto de prueba:")
    y_pred_test = predict_decision_tree(model, X_test_with_clusters)
    evaluate_model(y_test, y_pred_test)
    
    # Predicción interactiva
    interactive_prediction(model, scaler, kmeans)

def interactive_prediction(model, scaler, kmeans):
    """
    Permite al usuario ingresar datos para predecir si un tumor es benigno o maligno.
    
    Parámetros:
    - model: El modelo de arbol de decision entrenado.
    - scaler: El escalador ajustado para normalizar los datos del usuario.
    - kmeans: El modelo K-Means entrenado para predecir el cluster del nuevo dato.
    
    lo que se devuelve es:
    - La prediccion sobre si el tumor ingresado es benigno o maligno.
    """
    print("\nIntroduce los valores del nuevo tumor para hacer una predicción:")
    
    # El usuario ingresa los valores de 10 caracteristicas principales
    inputs = []
    features = ['Mean Radius', 'Mean Texture', 'Mean Perimeter', 'Mean Area', 
                'Mean Smoothness', 'Mean Compactness', 'Mean Concavity', 
                'Mean Concave Points', 'Mean Symmetry', 'Mean Fractal Dimension']
    for feature in features:
        value = float(input(f"{feature}: "))
        inputs.append(value)
    
    # Se generan valores para las otras 20 características de forma predeterminada
    std_error = [value * 0.1 for value in inputs]  # Suponer que el 'error estándar' es el 10% del valor
    worst = [value * 1.2 for value in inputs]  # Suponer que el 'peor' valor es un 20% mayor que la media

    # Aqui se Combina las 30 características: [media, error estándar, peor]
    new_data = np.array([inputs + std_error + worst])
    
    # escalamos los datos del nuevo paciente utilizando el scaler
    new_data_scaled = scaler.transform(new_data)
    
    # Se usa el modelo K-Means ya entrenado para predecir el cluster del nuevo dato
    new_cluster = kmeans.predict(new_data_scaled)
    new_data_with_cluster = np.hstack((new_data_scaled, new_cluster.reshape(-1, 1)))
    
    # Hacer la prediccion usando el arbol de decision
    prediction = predict_decision_tree(model, new_data_with_cluster)
    
    # Mostrar el resultado de la prediccion
    if prediction[0] == 1:
        print("\nEl tumor probablemente es maligno (Predicción: Maligno).")
    else:
        print("\nEl tumor probablemente es benigno (Predicción: Benigno).")

if __name__ == "__main__":
    main()