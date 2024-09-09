from sklearn.tree import DecisionTreeClassifier

def train_decision_tree(X_train, y_train):
    """
    Se entrena un modelo de arbol de Decision con los datos de entrenamiento.
    
    y regresa:
    - model: El modelo de arbol de decisión entrenado.
    """
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def predict_decision_tree(model, X_test):
    """
    YA aqui realiza predicciones usando un arbol de decisión entrenado.
    
    Retorna:
    - Predicciones realizadas por el modelo.
    """
    return model.predict(X_test)