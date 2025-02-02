import numpy as np
import pandas as pd



def predecir_energia_fotovoltaica(df, modelo, orden_columnas):
    """
    Predice la presencia de energía fotovoltaica en viviendas utilizando un modelo de Machine Learning.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con las viviendas sobre las que se realizará la predicción.
        Debe contener todas las características utilizadas en el entrenamiento del modelo.

    modelo : sklearn.base.BaseEstimator
        Modelo de Machine Learning previamente entrenado, como Random Forest o XGBoost,
        que predice la presencia de energía fotovoltaica.

    orden_columnas : list
        Lista con el orden correcto de las columnas que el modelo espera como entrada.
        Debe coincidir exactamente con las variables utilizadas en el entrenamiento.

    Retorna:
    --------
    pd.DataFrame
        DataFrame con las mismas columnas de entrada, pero añadiendo la columna `energia_fotovoltaica` 
        con las predicciones del modelo.

    Notas:
    ------
    - Se elimina la columna `"valor"` antes de realizar la predicción.
    - Se imprime la lista de columnas del DataFrame y las esperadas por el modelo para verificar consistencia.
    - Se imprime la distribución de las predicciones (`value_counts()`) para un análisis rápido de los resultados.

    Ejemplo:
    --------
    >>> df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4], 'valor': [100, 200]})
    >>> modelo = RandomForestClassifier().fit(X_train, y_train)
    >>> orden_columnas = ['col1', 'col2']
    >>> resultado = predecir_energia_fotovoltaica(df, modelo, orden_columnas)
    Columnas del DataFrame actual:
     Index(['col1', 'col2'], dtype='object')

    Columnas utilizadas en el modelo:
     ['col1', 'col2']

    -------------------------------------------------------------------------------

    Distribución de la variable 'energia_fotovoltaica':
    1    1
    0    1
    Name: energia_fotovoltaica, dtype: int64
    """
    # elimino la columna 'valor'
    df_sin_val = df.drop(columns=["valor"])
    
    # Confirmo que el modelo se entrenó con las mismas variables
    print("Columnas del DataFrame actual:\n", df_sin_val.columns)
    print("\nColumnas utilizadas en el modelo:\n", modelo.feature_names_in_)
    print(f"\n-------------------------------------------------------------------------------\n")
    # Las ordeno para coincidir con el entrenamiento del modelo
    df_sin_val = df_sin_val[orden_columnas]
    
    # Hago la predicción
    df_sin_val["energia_fotovoltaica"] = modelo.predict(df_sin_val)
    
    
    # Hago un value counts
    print("\nDistribución de la variable 'energia_fotovoltaica':\n")
    print(df_sin_val["energia_fotovoltaica"].value_counts())
    
    return df_sin_val