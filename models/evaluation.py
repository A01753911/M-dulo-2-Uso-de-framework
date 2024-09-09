from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(y_test, y_pred):
    """
    Aqui se evalua el rendimiento del modelo calculando la precision y generando una matriz de confusion.
    
    Devuelve las metricas:
    - Precisión (accuracy) del modelo.
    - Matriz de confusion (impresa y graficada).
    - Reporte de clasificacion con metricas de precision, recall y f1-score.
    """
    accuracy = (y_pred == y_test).mean()
    
    # Crear la matriz de confusión
    cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nMatriz de confusión:")
    print(cm)
    
    # Graficar la matriz de confusión usando un mapa de calor
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.show()
    
    # Obtener el reporte de clasificación como diccionario
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Imprimir el reporte de clasificación en el formato deseado
    print("\nReporte de clasificación:\n")
    print("                 Precisión   Recall   F1-Score")
    print(f"           0       {report['0']['precision']:.2f}      {report['0']['recall']:.2f}      {report['0']['f1-score']:.2f}")
    print(f"           1       {report['1']['precision']:.2f}      {report['1']['recall']:.2f}      {report['1']['f1-score']:.2f}")
    print(f"\n    accuracy                           {accuracy:.2f}")
