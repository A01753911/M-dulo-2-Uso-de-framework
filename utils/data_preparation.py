import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_data(filepath):
    """
    Prepara los datos para el modelo, incluyendo limpieza y normalizacion.
    
    lo que devuelve es:
    - X_scaled: datos normalizados.
    - y: Etiquetas objetivo (0 para benigno, 1 para maligno).
    - scaler: Escalador ajustado para normalizar nuevos datos.
    """
    df = pd.read_csv(filepath)
    
    # Tenemos una columna sin datos entonces la columna innecesaria la quitamos la cual es 'Unnamed: 32'
    df = df.drop(columns=['Unnamed: 32'])
    
    # para pasar el dato a booleano, la columna 'diagnosis' la pasamos en binario (1: maligno, 0: benigno)
    df['diagnosis'] = df['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)
    
    # Y aqui ya separamos características (X) y objetivo (y)
    X = df.iloc[:, 2:].values  # Omitimos las dos primeras columnas (ID, diagnosis)
    y = df['diagnosis'].values
    
    # Reemplazamos los valores NaN por la mediana
    X = pd.DataFrame(X).fillna(pd.DataFrame(X).median()).values
    
    # se normalizan los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def split_data(X, y, train_size=0.7, validation_size=0.15):
    """
    Divide los datos en conjuntos de entrenamiento, validación y prueba.
        
    devuelve lo siguiente:
    - X_train, X_validation, X_test: caracteristicas de cada conjunto.
    - y_train, y_validation, y_test: etiquetas de cada conjunto.
    """
    m = len(y)
    indices = np.arange(m)
    np.random.shuffle(indices)
    
    train_end = int(train_size * m)
    validation_end = int(validation_size * m) + train_end
    
    X_train = X[indices[:train_end]]
    y_train = y[indices[:train_end]]
    
    X_validation = X[indices[train_end:validation_end]]
    y_validation = y[indices[train_end:validation_end]]
    
    X_test = X[indices[validation_end:]]
    y_test = y[indices[validation_end:]]
    
    return X_train, X_validation, X_test, y_train, y_validation, y_test
