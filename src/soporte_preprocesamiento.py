# Tratamiento de datos
# -----------------------------------------------------------------------
import numpy as np
import pandas as pd

import re
# Otros objetivos
# -----------------------------------------------------------------------
import math

# Gráficos
# -----------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency
# Para tratar el problema de desbalance
# -----------------------------------------------------------------------
from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek

import os
import sys 

import warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath("src"))   
import soporte_preprocesamiento as f

# -----------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix ,roc_auc_score, cohen_kappa_score


#EDA

def exploracion_dataframe(dataframe, columna_control, estadisticos = False):
    """
    Realiza un análisis exploratorio básico de un DataFrame, mostrando información sobre duplicados,
    valores nulos, tipos de datos, valores únicos para columnas categóricas y estadísticas descriptivas
    para columnas categóricas y numéricas, agrupadas por la columna de control.

    Params:
    - dataframe (DataFrame): El DataFrame que se va a explorar.
    - columna_control (str): El nombre de la columna que se utilizará como control para dividir el DataFrame.

    Returns: 
    No devuelve nada directamente, pero imprime en la consola la información exploratoria.
    """
    print(f"El número de datos es {dataframe.shape[0]} y el de columnas es {dataframe.shape[1]}")
    print("\n ..................... \n")

    print(f"Los duplicados que tenemos en el conjunto de datos son: {dataframe.duplicated().sum()}")
    print("\n ..................... \n")
    
    
    # generamos un DataFrame para los valores nulos
    print("Los nulos que tenemos en el conjunto de datos son:")
    df_nulos = pd.DataFrame(dataframe.isnull().sum() / dataframe.shape[0] * 100, columns = ["%_nulos"])
    display(df_nulos[df_nulos["%_nulos"] > 0])
    
    # Tipos de columnas
    print("\n ..................... \n")
    print(f"Los tipos de las columnas son:")
    display(pd.DataFrame(dataframe.dtypes, columns = ["tipo_dato"]))
    
    # Enseñar solo las columnas categoricas (o tipo objeto)
    print("\n ..................... \n")
    print("Los valores que tenemos para las columnas categóricas son: ")
    dataframe_categoricas = dataframe.select_dtypes(include = "O")
    
    for col in dataframe_categoricas.columns:
        print(f"La columna {col.upper()} tiene los siguientes valores únicos:")
        print(f"Mostrando {pd.DataFrame(dataframe[col].value_counts()).head().shape[0]} categorías con más valores del total de {len(pd.DataFrame(dataframe[col].value_counts()))} categorías ({pd.DataFrame(dataframe[col].value_counts()).head().shape[0]}/{len(pd.DataFrame(dataframe[col].value_counts()))})")
        display(pd.DataFrame(dataframe[col].value_counts()).head())    
    
    # como estamos en un problema de A/B testing y lo que realmente nos importa es comparar entre el grupo de control y el de test, los principales estadísticos los vamos a sacar de cada una de las categorías
    if estadisticos == True:
        for categoria in dataframe[columna_control].unique():
            dataframe_filtrado = dataframe[dataframe[columna_control] == categoria]
            #Describe de objetos
            print("\n ..................... \n")

            print(f"Los principales estadísticos de las columnas categóricas para el {categoria.upper()} son: ")
            display(dataframe_filtrado.describe(include = "O").T)

            #Hacer un describe
            print("\n ..................... \n")
            print(f"Los principales estadísticos de las columnas numéricas para el {categoria.upper()} son: ")
            display(dataframe_filtrado.describe().T)
    else: 
        pass
    print("\n----------\n")
    print("Las principales estadísticas de las variables númericas son:")
    display(dataframe.describe().T)

    print("\n----------\n")
    print("Las principales estadísticas de las variables categóricas son:")
    display(dataframe.describe(include = "O").T)

    print("\n----------\n")
    print("Las características principales del dataframe son:")
    display(dataframe.info())

def convertir_a_cat(lista_cols_convertir,df):
    for col in lista_cols_convertir:
        df[col]=df[col].astype("category")
        
    return df    


def obtener_categoria(intervalo):
    """
    Clasifica un intervalo de metros cuadrados en una categoría de tamaño de vivienda.

    Parámetros:
    -----------
    intervalo : str
        Una cadena que representa un rango de metros cuadrados en formato de texto.
        Por ejemplo: "De 46 a 60", "De 91 a 105", etc.

    Retorna:
    --------
    str
        Una categoría que clasifica el tamaño de la vivienda según el intervalo proporcionado.
        Las categorías posibles son:
        - "Viviendas pequeñas": Si el máximo del intervalo es menor o igual a 50.
        - "Viviendas medianas": Si el máximo del intervalo está entre 51 y 90.
        - "Viviendas amplias": Si el máximo del intervalo está entre 91 y 150.
        - "Viviendas grandes": Si el mínimo del intervalo es 180 o si el máximo supera 150.

    Notas:
    ------
    - El intervalo debe estar en un formato que contenga números, como "De 31 a 45".
    - Se utiliza una expresión regular para extraer los valores numéricos del intervalo.

    Ejemplo:
    --------
    >>> obtener_categoria("De 31 a 45")
    'Viviendas pequeñas'

    >>> obtener_categoria("De 91 a 105")
    'Viviendas amplias'

    >>> obtener_categoria("De 151 a 180")
    'Viviendas grandes'
    """
    # consigo los valores numéricos del intervalo
    match = re.findall(r'\d+', intervalo)
    minimo = int(match[0])
    maximo = int(match[1])
    # los clasifico en cada categoría
    if minimo== 180:
        return "Viviendas grandes"
    elif  maximo <= 50:
        return "Viviendas pequeñas"
    elif 50 < maximo <= 90:
        return "Viviendas medianas"
    elif 90 < maximo <= 150:
        return "Viviendas amplias"
    else:
        return "Viviendas grandes"




def formatear_provincia(provincia):
    """
    Formatea el nombre de una provincia para asegurar consistencia en su presentación.

    Args:
        provincia (str): Nombre de la provincia a formatear.

    Returns:
        str: Nombre de la provincia formateado con la primera letra en mayúscula y
             reemplazos específicos para nombres especiales.

    Funcionalidad:
        - Convierte el nombre de la provincia a formato "Title Case" (primera letra en mayúscula).
        - Realiza un reemplazo específico para "Tenerife", devolviendo "Santa Cruz de Tenerife".

    Ejemplo:
        ```python
        print(formatear_provincia("tenerife"))  # Output: "Santa Cruz de Tenerife"
        print(formatear_provincia("madrid"))    # Output: "Madrid"
        ```
    """
    provincia = provincia.title()  # Capitalizar la primera letra
    if provincia == "Tenerife":
        return "Santa Cruz de Tenerife"  # Reemplazo específico
    return provincia

def categorizar_metros(metrosvi):
    """
    Categoriza un valor numérico de metros cuadrados en intervalos predefinidos o asigna 'Desconocido' si corresponde.

    Parámetros:
    -----------
    metrosvi : float o str
        Valor numérico que representa los metros cuadrados o 'Desconocido'.

    Retorna:
    --------
    str
        Una cadena que indica el intervalo al que pertenece el valor de metros cuadrados, según las siguientes categorías:
        - "Hasta 30 m2"
        - "Entre 31 y 45 m2"
        - "Entre 46 y 60 m2"
        - "Entre 61 y 75 m2"
        - "Entre 76 y 90 m2"
        - "Entre 91 y 105 m2"
        - "Entre 106 y 120 m2"
        - "Entre 121 y 150 m2"
        - "Entre 151 y 180 m2"
        - "Más de 180 m2"
        - "Desconocido" (si el valor de entrada es 'Desconocido')

    Ejemplos:
    ---------
    >>> categorizar_metros(25)
    'Hasta 30 m2'

    >>> categorizar_metros(50)
    'Entre 46 y 60 m2'

    >>> categorizar_metros("Desconocido")
    'Desconocido'
    """
    if metrosvi == "Desconocido":
        return "Desconocido"
    elif metrosvi <= 30:
        return "Hasta 30 m2"
    elif 31 <= metrosvi <= 45:
        return "Entre 31 y 45 m2"
    elif 46 <= metrosvi <= 60:
        return "Entre 46 y 60 m2"
    elif 61 <= metrosvi <= 75:
        return "Entre 61 y 75 m2"
    elif 76 <= metrosvi <= 90:
        return "Entre 76 y 90 m2"
    elif 91 <= metrosvi <= 105:
        return "Entre 91 y 105 m2"
    elif 106 <= metrosvi <= 120:
        return "Entre 106 y 120 m2"
    elif 121 <= metrosvi <= 150:
        return "Entre 121 y 150 m2"
    elif 151 <= metrosvi <= 180:
        return "Entre 151 y 180 m2"
    else:
        return "Más de 180 m2"



def categorizar_garajes(garajes):
    """
    Categoriza el número de plazas de garaje en intervalos predefinidos.

    Esta función toma como entrada un valor numérico de plazas de garaje (o NaN) y devuelve una categoría textual según los siguientes criterios:
    - Si el valor es NaN: "No tiene garaje".
    - Si el valor es 1: "1".
    - Si el valor es 2: "2".
    - Si el valor está entre 3 y 5 (inclusive): "De 3 a 5".
    - Si el valor está entre 6 y 10 (inclusive): "De 6 a 10".
    - Si el valor está entre 11 y 20 (inclusive): "De 11 a 20".
    - Si el valor está entre 21 y 50 (inclusive): "De 21 a 50".
    - Si el valor está entre 51 y 100 (inclusive): "De 51 a 100".
    - Si el valor está entre 101 y 150 (inclusive): "De 101 a 150".
    - Si el valor es mayor a 150: "Más de 150".

    Args:
        garajes (float or int): Número de plazas de garaje. Puede ser un valor numérico o NaN.

    Returns:
        str: Categoría textual que describe el número de plazas de garaje.
    """
    if pd.isna(garajes):
        return "No tiene garaje"
    elif 1 <= garajes <= 1:
        return "1"
    elif 2 <= garajes <= 2:
        return "2"
    elif 3 <= garajes <= 5:
        return "De 3 a 5"
    elif 6 <= garajes <= 10:
        return "De 6 a 10"
    elif 11 <= garajes <= 20:
        return "De 11 a 20"
    elif 21 <= garajes <= 50:
        return "De 21 a 50"
    elif 51 <= garajes <= 100:
        return "De 51 a 100"
    elif 101 <= garajes <= 150:
        return "De 101 a 150"
    else:
        return "Más de 150"



def asignar_intervalo(anio):
    """
    Asigna un intervalo de años a partir de un valor numérico que representa un año de construcción.

    Parámetros:
    -----------
    anio : int o float
        El año de construcción a clasificar en intervalos. 
        Puede ser un valor numérico o NaN (Not a Number).

    Retorna:
    --------
    str
        Una cadena que indica el intervalo correspondiente al año proporcionado.
        Las categorías incluyen:
        - 'Desconocido': Si el valor es NaN.
        - 'Anterior a 1900': Si el año es menor o igual a 1900.
        - 'De 1900 a 1920': Si el año está entre 1901 y 1920.
        - 'De 1921 a 1940': Si el año está entre 1921 y 1940.
        - 'De 1941 a 1950': Si el año está entre 1941 y 1950.
        - 'De 1951 a 1960': Si el año está entre 1951 y 1960.
        - 'De 1961 a 1970': Si el año está entre 1961 y 1970.
        - 'De 1971 a 1980': Si el año está entre 1971 y 1980.
        - 'De 1981 a 1990': Si el año está entre 1981 y 1990.
        - 'De 1991 a 2000': Si el año está entre 1991 y 2000.
        - 'De 2001 a 2010': Si el año está entre 2001 y 2010.
        - 'Posterior a 2010': Si el año es mayor a 2010.

    Ejemplo:
    --------
    >>> asignar_intervalo(1895)
    'Anterior a 1900'

    >>> asignar_intervalo(1955)
    'De 1951 a 1960'

    >>> asignar_intervalo(2020)
    'Posterior a 2010'

    >>> asignar_intervalo(float('nan'))
    'Desconocido'
    """
    if pd.isna(anio):
        return 'Desconocido'
    elif anio <= 1900:
        return 'Anterior a  1900'
    elif 1901 <= anio <= 1920:
        return 'De 1900 a 1920'
    elif 1921 <= anio <= 1940:
        return 'De 1921 a 1940'
    elif 1941 <= anio <= 1950:
        return 'De 1941 a 1950'
    elif 1951 <= anio <= 1960:
        return 'De 1951 a 1960'
    elif 1961 <= anio <= 1970:
        return 'De 1961 a 1970'
    elif 1971 <= anio <= 1980:
        return 'De 1971 a 1980'
    elif 1981 <= anio <= 1990:
        return 'De 1981 a 1990'
    elif 1991 <= anio <= 2000:
        return 'De 1991 a 2000'
    elif 2001 <= anio <= 2010:
        return 'De 2001 a 2010'
    else:
        return 'Posterior a 2010' 




def calcular_anio_construccion(antiguedad):
    """
    Calcula un año aproximado de construcción basado en categorías de antigüedad.

    Parámetros:
    -----------
    antiguedad : str
        Una cadena que representa la categoría de antigüedad de un inmueble.
        Las categorías válidas incluyen:
        - 'Menos de 5 Años'
        - 'Entre 5 y 10 Años'
        - 'Entre 10 y 15 Años'
        - 'Entre 15 y 20 Años'
        - 'Entre 20 y 25 Años'
        - 'Entre 25 y 35 años'
        - 'Entre 35 y 50 Años'
        - 'Mas de 25 Años' (una categoría adicional que usa un promedio dentro del rango)
        - 'Más de 50 años'

    Retorna:
    --------
    int o numpy.nan
        El año aproximado de construcción calculado a partir de la antigüedad:
        - Para cada categoría, se resta el límite superior del rango al año actual (2024).
        - Si la antigüedad no coincide con ninguna categoría conocida, devuelve numpy.nan.

    Ejemplo:
    --------
    >>> calcular_anio_construccion('Menos de 5 Años')
    2019

    >>> calcular_anio_construccion('Entre 10 y 15 Años')
    2009

    >>> calcular_anio_construccion('Más de 50 años')
    1964

    >>> calcular_anio_construccion('Categoría inválida')
    nan
    """
    if antiguedad == 'Menos de 5 Años':
        return 2024 - 5
    elif antiguedad == 'Entre 5 y 10 Años':
        return 2024 - 10
    elif antiguedad == 'Entre 10 y 15 Años':
        return 2024 - 15
    elif antiguedad == 'Entre 15 y 20 Años':
        return 2024 - 20
    elif antiguedad == 'Entre 20 y 25 Años':
        return 2024 - 25
    elif antiguedad == 'Entre 25 y 35 años':
        return 2024 - 35
    elif antiguedad == 'Entre 35 y 50 Años':
        return 2024 - 50
    elif antiguedad == 'Mas de 25 Años':
        return 2024 - 30  # Usamos un promedio dentro del rango
    elif antiguedad == 'Más de 50 años':
        return 2024 - 60  # Suposición para valores superiores a 50 años
    else:
        return np.nan


class Visualizador:
    """
    Clase para visualizar la distribución de variables numéricas y categóricas de un DataFrame.

    Attributes:
    - dataframe (pandas.DataFrame): El DataFrame que contiene las variables a visualizar.

    Methods:
    - __init__: Inicializa el VisualizadorDistribucion con un DataFrame y un color opcional para las gráficas.
    - separar_dataframes: Separa el DataFrame en dos subconjuntos, uno para variables numéricas y otro para variables categóricas.
    - plot_numericas: Grafica la distribución de las variables numéricas del DataFrame.
    - plot_categoricas: Grafica la distribución de las variables categóricas del DataFrame.
    - plot_relacion2: Visualiza la relación entre una variable y todas las demás, incluyendo variables numéricas y categóricas.
    """

    def __init__(self, dataframe):
        """
        Inicializa el VisualizadorDistribucion con un DataFrame y un color opcional para las gráficas.

        Parameters:
        - dataframe (pandas.DataFrame): El DataFrame que contiene las variables a visualizar.
        - color (str, opcional): El color a utilizar en las gráficas. Por defecto es "grey".
        """
        self.dataframe = dataframe

    def separar_dataframes(self):
        """
        Separa el DataFrame en dos subconjuntos, uno para variables numéricas y otro para variables categóricas.

        Returns:
        - pandas.DataFrame: DataFrame con variables numéricas.
        - pandas.DataFrame: DataFrame con variables categóricas.
        """
        return self.dataframe.select_dtypes(include=np.number), self.dataframe.select_dtypes(include=["O", "category"])
    
    def plot_numericas(self, color="grey", tamano_grafica=(20, 10)):
        """
        Grafica la distribución de las variables numéricas del DataFrame.

        Parameters:
        - color (str, opcional): El color a utilizar en las gráficas. Por defecto es "grey".
        - tamaño_grafica (tuple, opcional): El tamaño de la figura de la gráfica. Por defecto es (15, 5).
        """
        lista_num = self.separar_dataframes()[0].columns
        fig, axes = plt.subplots(ncols = 2, nrows = math.ceil(len(lista_num)/2), figsize=tamano_grafica, sharey=True)
        axes = axes.flat
        for indice, columna in enumerate(lista_num):
            sns.histplot(x=columna, data=self.dataframe, ax=axes[indice], color=color, bins=20)
            axes[indice].set_title(f"Distribución de {columna}")
        plt.suptitle("Distribución de variables numéricas")
        plt.tight_layout()

        if len(lista_num) % 2 !=0:
            fig.delaxes(axes[-1])


    def plot_categoricas(self, color="grey", tamanio_grafica=(20, 10), tamanio_fuente=14):
        """
        Grafica la distribución de las variables categóricas del DataFrame.

        Parameters:
        - color (str, opcional): El color a utilizar en las gráficas. Por defecto es "grey".
        - tamaño_grafica (tuple, opcional): El tamaño de la figura de la gráfica. Por defecto es (15, 5).
        """
        lista_cat = self.separar_dataframes()[1].columns
        fig, axes = plt.subplots(ncols = 2, nrows = math.ceil(len(lista_cat) / 2), figsize=tamanio_grafica)
        axes = axes.flat
        for indice, columna in enumerate(lista_cat):
            sns.countplot(x=columna, data=self.dataframe, order=self.dataframe[columna].value_counts().index,
                          ax=axes[indice], color=color)
            axes[indice].tick_params(rotation=90, labelsize=tamanio_fuente)
            axes[indice].set_title(columna, fontsize=tamanio_fuente+2)
            axes[indice].set(xlabel=None)
            plt.tight_layout()

        plt.suptitle("Distribución de variables categóricas", fontsize=tamanio_fuente + 5)
        plt.tight_layout()

        if len(lista_cat) % 2 !=0:
            fig.delaxes(axes[-1])


    def plot_relacion(self, vr, tamano_grafica=(20, 10), tamanio_fuente=18):
        """
        Genera gráficos que muestran la relación entre cada columna del DataFrame y una variable de referencia (vr).
        Los gráficos son adaptativos según el tipo de dato: histogramas para variables numéricas y countplots para categóricas.

        Parámetros:
        -----------
        vr : str
            Nombre de la columna que actúa como la variable de referencia para las relaciones.
        tamano_grafica : tuple, opcional
            Tamaño de la figura en el formato (ancho, alto). Por defecto es (20, 10).
        tamanio_fuente : int, opcional
            Tamaño de la fuente para los títulos de los gráficos. Por defecto es 18.

        Retorno:
        --------
        None
            Muestra una serie de subgráficos con las relaciones entre la variable de referencia y el resto de columnas del DataFrame.

        Notas:
        ------
        - La función asume que el DataFrame de interés está definido dentro de la clase como `self.dataframe`.
        - Se utiliza `self.separar_dataframes()` para obtener las columnas numéricas y categóricas en listas separadas.
        - La variable de referencia (`vr`) no será graficada contra sí misma.
        - Los gráficos utilizan la paleta "magma" para la diferenciación de categorías o valores de la variable de referencia.
        """

        lista_num = self.separar_dataframes()[0].columns
        lista_cat = self.separar_dataframes()[1].columns

        fig, axes = plt.subplots(ncols = 2, nrows = math.ceil(len(self.dataframe.columns) / 2), figsize=tamano_grafica)
        axes = axes.flat

        for indice, columna in enumerate(self.dataframe.columns):
            if columna == vr:
                fig.delaxes(axes[indice])
            elif columna in lista_num:
                sns.histplot(x = columna, 
                             hue = vr, 
                             data = self.dataframe, 
                             ax = axes[indice], 
                             palette = "magma", 
                             legend = False)
                plt.tight_layout()
                
            elif columna in lista_cat:
                sns.countplot(x = columna, 
                              hue = vr, 
                              data = self.dataframe, 
                              ax = axes[indice], 
                              palette = "magma"
                              )
                plt.tight_layout()

            axes[indice].set_title(f"Relación {columna} vs {vr}",size=tamanio_fuente)   

        plt.tight_layout()
    


    def plot_relacion_individual(self, vr, tamano_grafica=(20, 10), tamanio_fuente=18):
        """
        Genera gráficos que muestran la relación entre cada columna del DataFrame y una variable de referencia (vr).
        Los gráficos se generan en subplots para las variables numéricas, categóricas con menos de 6 categorías y categóricas con 6 o más categorías.

        Parámetros:
        -----------
        vr : str
            Nombre de la columna que actúa como la variable de referencia para las relaciones.
        tamano_grafica : tuple, opcional
            Tamaño de la figura en el formato (ancho, alto). Por defecto es (20, 10).
        tamanio_fuente : int, opcional
            Tamaño de la fuente para los títulos de los gráficos. Por defecto es 18.

        Retorno:
        --------
        None
            Muestra una serie de gráficos para cada columna con vr=0 y vr=1.
        """


        lista_num = self.separar_dataframes()[0].columns
        lista_cat = self.separar_dataframes()[1].columns

        
        df_vr_1 = self.dataframe[self.dataframe[vr] == 1]
        df_vr_0 = self.dataframe[self.dataframe[vr] == 0]

        # variables numéricas
        num_vars = [col for col in lista_num if col != vr]
        fig_num, axes_num = plt.subplots(ncols=2, nrows=len(num_vars), figsize=tamano_grafica)
        axes_num = axes_num.flat

        for i, columna in enumerate(num_vars):
            
            sns.histplot(x=columna, data=df_vr_0, ax=axes_num[i * 2], color="blue", kde=False)
            axes_num[i * 2].set_title(f"{columna} (VR=0)", size=tamanio_fuente)

            
            sns.histplot(x=columna, data=df_vr_1, ax=axes_num[i * 2 + 1], color="orange", kde=False)
            axes_num[i * 2 + 1].set_title(f"{columna} (VR=1)", size=tamanio_fuente)

            
            x_min = min(self.dataframe[columna].min(), df_vr_0[columna].min(), df_vr_1[columna].min())
            x_max = max(self.dataframe[columna].max(), df_vr_0[columna].max(), df_vr_1[columna].max())
            x_range = x_max-x_min

            x_min -= x_range*0.05
            x_max += x_range*0.05

            axes_num[i * 2].set_xlim(x_min, x_max)
            axes_num[i * 2 + 1].set_xlim(x_min, x_max)

        plt.tight_layout()
        plt.suptitle("Variables Numéricas", fontsize=tamanio_fuente + 2)
        plt.show()

        # variables categóricas con menos de 6 categorías
        cat_vars_small = [col for col in lista_cat if col != vr and self.dataframe[col].nunique() < 6]
        fig_cat_small, axes_cat_small = plt.subplots(ncols=2, nrows=len(cat_vars_small), figsize=tamano_grafica)
        axes_cat_small = axes_cat_small.flat

        for i, columna in enumerate(cat_vars_small):
            
            orden = self.dataframe[columna].value_counts().index

            sns.countplot(x=columna, data=df_vr_0, ax=axes_cat_small[i * 2], order=orden, color="blue")
            axes_cat_small[i * 2].set_title(f"{columna} (VR=0)", size=tamanio_fuente)

            sns.countplot(x=columna, data=df_vr_1, ax=axes_cat_small[i * 2 + 1], order=orden, color="orange")
            axes_cat_small[i * 2 + 1].set_title(f"{columna} (VR=1)", size=tamanio_fuente)

        plt.tight_layout()
        plt.suptitle("Variables Categóricas (<6 Categorías)", fontsize=tamanio_fuente + 2)
        plt.show()

        # variables categóricas con 6 o más categorías
        cat_vars_large = [col for col in lista_cat if col != vr and self.dataframe[col].nunique() >= 6]
        for columna in cat_vars_large:
            fig_cat_large, axes_cat_large = plt.subplots(nrows=1, ncols=2, figsize=(tamano_grafica[0], tamano_grafica[1] // 2))

            orden = self.dataframe[columna].value_counts().index

            sns.countplot(x=columna, data=df_vr_0, ax=axes_cat_large[0], order=orden, color="blue")
            axes_cat_large[0].set_title(f"{columna} (VR=0)", size=tamanio_fuente)
            axes_cat_large[0].tick_params(axis='x', rotation=90)

            sns.countplot(x=columna, data=df_vr_1, ax=axes_cat_large[1], order=orden, color="orange")
            axes_cat_large[1].set_title(f"{columna} (VR=1)", size=tamanio_fuente)
            axes_cat_large[1].tick_params(axis='x', rotation=90)

            plt.tight_layout()
            plt.suptitle(f"{columna} (>=6 Categorías)", fontsize=tamanio_fuente + 2)
            plt.show()



    def plot_relacion_individual_per(self, vr, tamano_grafica=(20, 10), tamanio_fuente=18):
        """
        Genera gráficos que muestran las proporciones (%) de las categorías dentro de los grupos VR=0 y VR=1
        para las variables categóricas y las numéricas del DataFrame.

        Parámetros:
        -----------
        vr : str
            Nombre de la columna que actúa como la variable de referencia para las relaciones.
        tamano_grafica : tuple, opcional
            Tamaño de la figura en el formato (ancho, alto). Por defecto es (20, 10).
        tamanio_fuente : int, opcional
            Tamaño de la fuente para los títulos de los gráficos. Por defecto es 18.

        Retorno:
        --------
        None
            Muestra una serie de gráficos para cada columna con VR=0 y VR=1.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        
        lista_num = self.separar_dataframes()[0].columns
        lista_cat = self.separar_dataframes()[1].columns

        
        df_vr_0 = self.dataframe[self.dataframe[vr] == 0]
        df_vr_1 = self.dataframe[self.dataframe[vr] == 1]

        #numéricas
        num_vars = [col for col in lista_num if col != vr]
        fig_num, axes_num = plt.subplots(ncols=2, nrows=len(num_vars), figsize=tamano_grafica)
        axes_num = axes_num.flat

        for i, columna in enumerate(num_vars):
            
            total_vr_0 = len(df_vr_0)
            total_vr_1 = len(df_vr_1)

            sns.histplot(
                x=columna, data=df_vr_0, ax=axes_num[i * 2], color="blue", kde=False, stat="percent",
                weights=[100 / total_vr_0] * total_vr_0
            )
            axes_num[i * 2].set_title(f"{columna} (VR=0)", size=tamanio_fuente)

            sns.histplot(
                x=columna, data=df_vr_1, ax=axes_num[i * 2 + 1], color="orange", kde=False, stat="percent",
                weights=[100 / total_vr_1] * total_vr_1
            )
            axes_num[i * 2 + 1].set_title(f"{columna} (VR=1)", size=tamanio_fuente)

        plt.tight_layout()
        plt.suptitle("Variables Numéricas (Proporciones)", fontsize=tamanio_fuente + 2)
        plt.show()

        #categóricas con menos de 6 categorías
        cat_vars_small = [col for col in lista_cat if col != vr and self.dataframe[col].nunique() < 6]
        fig_cat_small, axes_cat_small = plt.subplots(ncols=2, nrows=len(cat_vars_small), figsize=tamano_grafica)
        axes_cat_small = axes_cat_small.flat

        for i, columna in enumerate(cat_vars_small):
            
            proporciones_vr_0 = df_vr_0[columna].value_counts(normalize=True) * 100
            proporciones_vr_1 = df_vr_1[columna].value_counts(normalize=True) * 100

            orden = self.dataframe[columna].value_counts().index

            sns.barplot(
                x=proporciones_vr_0.index, y=proporciones_vr_0.values, ax=axes_cat_small[i * 2], order=orden, color="blue"
            )
            axes_cat_small[i * 2].set_title(f"{columna} (VR=0)", size=tamanio_fuente)
            axes_cat_small[i * 2].set_ylabel("% dentro de VR=0")

            sns.barplot(
                x=proporciones_vr_1.index, y=proporciones_vr_1.values, ax=axes_cat_small[i * 2 + 1], order=orden, color="orange"
            )
            axes_cat_small[i * 2 + 1].set_title(f"{columna} (VR=1)", size=tamanio_fuente)
            axes_cat_small[i * 2 + 1].set_ylabel("% dentro de VR=1")

        plt.tight_layout()
        plt.suptitle("Variables Categóricas (<6 Categorías, Proporciones)", fontsize=tamanio_fuente + 2)
        plt.show()

        #categóricas con 6 o más categorías
        cat_vars_large = [col for col in lista_cat if col != vr and self.dataframe[col].nunique() >= 6]
        for columna in cat_vars_large:
            fig_cat_large, axes_cat_large = plt.subplots(nrows=1, ncols=2, figsize=(tamano_grafica[0], tamano_grafica[1] // 2))

            proporciones_vr_0 = df_vr_0[columna].value_counts(normalize=True) * 100
            proporciones_vr_1 = df_vr_1[columna].value_counts(normalize=True) * 100

            orden = self.dataframe[columna].value_counts().index

            sns.barplot(
                x=proporciones_vr_0.index, y=proporciones_vr_0.values, ax=axes_cat_large[0], order=orden, color="blue"
            )
            axes_cat_large[0].set_title(f"{columna} (VR=0)", size=tamanio_fuente)
            axes_cat_large[0].tick_params(axis="x", rotation=90)
            axes_cat_large[0].set_ylabel("% dentro de VR=0")

            sns.barplot(
                x=proporciones_vr_1.index, y=proporciones_vr_1.values, ax=axes_cat_large[1], order=orden, color="orange"
            )
            axes_cat_large[1].set_title(f"{columna} (VR=1)", size=tamanio_fuente)
            axes_cat_large[1].tick_params(axis="x", rotation=90)
            axes_cat_large[1].set_ylabel("% dentro de VR=1")

            plt.tight_layout()
            plt.suptitle(f"{columna} (>=6 Categorías, Proporciones)", fontsize=tamanio_fuente + 2)
            plt.show()

        
    def deteccion_outliers(self, color = "grey", tamano_grafica = (20, 10)):

        """
        Detecta y visualiza valores atípicos en un DataFrame.

        Params:
            - dataframe (pandas.DataFrame):  El DataFrame que se va a usar

        Returns:
            No devuelve nada

        Esta función selecciona las columnas numéricas del DataFrame dado y crea un diagrama de caja para cada una de ellas para visualizar los valores atípicos.
        """

        lista_num = self.separar_dataframes()[0].columns

        fig, axes = plt.subplots(ncols = 2, nrows = math.ceil(len(lista_num)/2), figsize=tamano_grafica)
        axes = axes.flat

        for indice, columna in enumerate(lista_num):
            sns.boxplot(x=columna, data=self.dataframe, 
                        ax=axes[indice], 
                        color=color, 
                        flierprops={'markersize': 4, 'markerfacecolor': 'orange'})
            axes[indice].set_title(f"Outliers {columna}")  

        if len(lista_num) % 2 != 0:
            fig.delaxes(axes[-1])

        
        plt.tight_layout()

    def correlacion(self, tamano_grafica = (7, 5)):

        """
        Visualiza la matriz de correlación de un DataFrame utilizando un mapa de calor.

        Params:
            - dataframe : pandas DataFrame. El DataFrame que contiene los datos para calcular la correlación.

        Returns:
        No devuelve nada 

        Muestra un mapa de calor de la matriz de correlación.

        - Utiliza la función `heatmap` de Seaborn para visualizar la matriz de correlación.
        - La matriz de correlación se calcula solo para las variables numéricas del DataFrame.
        - La mitad inferior del mapa de calor está oculta para una mejor visualización.
        - Permite guardar la imagen del mapa de calor como un archivo .png si se solicita.

        """

        plt.figure(figsize = tamano_grafica )

        mask = np.triu(np.ones_like(self.dataframe.corr(numeric_only=True), dtype = np.bool_))

        sns.heatmap(data = self.dataframe.corr(numeric_only = True), 
                    annot = True, 
                    vmin=-1,
                    vmax=1,
                    cmap="magma",
                    linecolor="black", 
                    fmt='.1g', 
                    mask = mask)
        
#ORDEN (ENCODING)

def detectar_orden_cat(df,lista_cat,var_respuesta):
    """
    Evalúa si las variables categóricas de una lista presentan un orden significativo en relación con una variable de respuesta,
    utilizando el test de Chi-cuadrado.

    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame que contiene las variables categóricas y la variable de respuesta.
    lista_cat : list
        Lista con los nombres de las columnas categóricas que se evaluarán.
    var_respuesta : str
        Nombre de la columna que actúa como la variable de respuesta.

    Retorno:
    --------
    None
        La función imprime:
        - Una tabla cruzada entre cada variable categórica y la variable de respuesta.
        - Un mensaje indicando si la variable categórica tiene un orden significativo en relación con la variable de respuesta.

    Notas:
    ------
    - El orden se evalúa mediante el p-value del test de Chi-cuadrado. Si `p < 0.05`, se concluye que la variable categórica tiene orden.
    - Se muestra cada tabla cruzada usando `display`, lo que es útil en entornos como Jupyter Notebook.
    """
    for categoria in lista_cat:
        print(f"Estamos evaluando el orden de la variable {categoria.upper()}")
        df_cross_tab=pd.crosstab(df[categoria], df[var_respuesta])
        display(df_cross_tab)
        
        chi2, p, dof, expected= chi2_contingency(df_cross_tab)

        if p <0.05:
            print(f"La variable {categoria} SI tiene orden porque su p-valor es : {p}")
        else:
            print(f"La variable {categoria} NO tiene orden porque su p-valor es : {p}")



class Encoding:
    """
    Clase para realizar diferentes tipos de codificación en un DataFrame.

    Parámetros:
        - dataframe: DataFrame de pandas, el conjunto de datos a codificar.
        - diccionario_encoding: dict, un diccionario que especifica los tipos de codificación a realizar.
        - variable_respuesta: str, el nombre de la variable objetivo.

    Métodos:
        - one_hot_encoding(): Realiza codificación one-hot en las columnas especificadas en el diccionario de codificación.
        - get_dummies(prefix='category', prefix_sep='_'): Realiza codificación get_dummies en las columnas especificadas en el diccionario de codificación.
        - ordinal_encoding(): Realiza codificación ordinal en las columnas especificadas en el diccionario de codificación.
        - label_encoding(): Realiza codificación label en las columnas especificadas en el diccionario de codificación.
        - target_encoding(): Realiza codificación target en la variable especificada en el diccionario de codificación.
        - frequency_encoding(): Realiza codificación de frecuencia en las columnas especificadas en el diccionario de codificación.
    """

    def __init__(self, dataframe, diccionario_encoding, variable_respuesta):
        self.dataframe = dataframe.copy()
        self.diccionario_encoding = diccionario_encoding
        self.variable_respuesta = variable_respuesta

    def one_hot_encoding(self):
        """
        Realiza codificación one-hot en las columnas especificadas en el diccionario de codificación.

        Returns:
            - dataframe: DataFrame de pandas, el DataFrame con codificación one-hot aplicada.
            - one_hot_encoder: El objeto OneHotEncoder utilizado para la codificación.
        """
        # Accedemos a la clave de 'onehot' para extraer las columnas a codificar.
        col_encode = self.diccionario_encoding.get("onehot", [])

        # Si hay columnas especificadas
        if col_encode:
            # Instanciamos OneHotEncoder
            one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

            # Aplicamos la codificación one-hot
            trans_one_hot = one_hot_encoder.fit_transform(self.dataframe[col_encode])

            # Convertimos el resultado a un DataFrame con nombres de columnas
            oh_df = pd.DataFrame(trans_one_hot, columns=one_hot_encoder.get_feature_names_out(col_encode))

            # Aseguramos que los índices coincidan antes de concatenar
            oh_df.index = self.dataframe.index

            # Concatenamos con el DataFrame original y eliminamos las columnas codificadas originales
            self.dataframe = pd.concat([self.dataframe.drop(columns=col_encode), oh_df], axis=1)

        return self.dataframe, one_hot_encoder
    
    def get_dummies(self, prefix='category', prefix_sep="_"):
        """
        Realiza codificación get_dummies en las columnas especificadas en el diccionario de codificación.

        Parámetros:
        - prefix: str, prefijo para los nombres de las nuevas columnas codificadas.
        - prefix_sep: str, separador entre el prefijo y el nombre original de la columna.

        Returns:
        - dataframe: DataFrame de pandas, el DataFrame con codificación get_dummies aplicada.
        """
        # accedemos a la clave de 'dummies' para poder extraer las columnas a las que que queramos aplicar este método. En caso de que no exista la clave, esta variable será una lista vacía
        col_encode = self.diccionario_encoding.get("dummies", [])

        if col_encode:
            # aplicamos el método get_dummies a todas las columnas seleccionadas, y determinamos su prefijo y la separación
            df_dummies = pd.get_dummies(self.dataframe[col_encode], dtype=int, prefix=prefix, prefix_sep=prefix_sep)
            
            # concatenamos los resultados del get_dummies con el DataFrame original
            self.dataframe = pd.concat([self.dataframe.reset_index(drop=True), df_dummies.reset_index(drop=True)], axis=1)
            
            # eliminamos las columnas original que ya no nos hacen falta
            self.dataframe.drop(col_encode, axis=1, inplace=True)
    
        return self.dataframe

    def ordinal_encoding(self):
        """
        Realiza codificación ordinal en las columnas especificadas en el diccionario de codificación.

        Returns:
        - dataframe: DataFrame de pandas, el DataFrame con codificación ordinal aplicada.
        """

        # Obtenemos las columnas a codificar bajo la clave 'ordinal'. Si no existe la clave, la variable col_encode será una lista vacía.
        col_encode = self.diccionario_encoding.get("ordinal", {})

        # Verificamos si hay columnas a codificar.
        if col_encode:

            # Obtenemos las categorías de cada columna especificada para la codificación ordinal.
            orden_categorias = list(self.diccionario_encoding["ordinal"].values())
            
            # Inicializamos el codificador ordinal con las categorías especificadas.
            ordinal_encoder = OrdinalEncoder(categories=orden_categorias, dtype=float, handle_unknown="use_encoded_value", unknown_value=np.nan)
            
            # Aplicamos la codificación ordinal a las columnas seleccionadas.
            ordinal_encoder_trans = ordinal_encoder.fit_transform(self.dataframe[col_encode.keys()])

            # Eliminamos las columnas originales del DataFrame.
            self.dataframe.drop(col_encode, axis=1, inplace=True)
            
            # Creamos un nuevo DataFrame con las columnas codificadas y sus nombres.
            ordinal_encoder_df = pd.DataFrame(ordinal_encoder_trans, columns=ordinal_encoder.get_feature_names_out())

            # Concatenamos el DataFrame original con el DataFrame de las columnas codificadas.
            self.dataframe = pd.concat([self.dataframe.reset_index(drop=True), ordinal_encoder_df], axis=1)

        return self.dataframe


    def label_encoding(self):
        """
        Realiza codificación label en las columnas especificadas en el diccionario de codificación.

        Returns:
        - dataframe: DataFrame de pandas, el DataFrame con codificación label aplicada.
        """

        # accedemos a la clave de 'label' para poder extraer las columnas a las que que queramos aplicar Label Encoding. En caso de que no exista la clave, esta variable será una lista vacía
        col_encode = self.diccionario_encoding.get("label", [])

        # si hay contenido en la lista 
        if col_encode:

            # instanciamos la clase LabelEncoder()
            label_encoder = LabelEncoder()

            # aplicamos el Label Encoder a cada una de las columnas, y creamos una columna con el nombre de la columna (la sobreescribimos)
            self.dataframe[col_encode] = self.dataframe[col_encode].apply(lambda col: label_encoder.fit_transform(col))
     
        return self.dataframe

    def target_encoding(self):
        """
        Realiza codificación target en la variable especificada en el diccionario de codificación.

        Returns:
        
        dataframe: DataFrame de pandas, el DataFrame con codificación target aplicada."""

        df_sin_vr = self.dataframe.copy()
        df_sin_vr.drop(columns=[f"{self.variable_respuesta}"], inplace=True)

        #accedemos a la clave de 'target' para poder extraer las columnas a las que que queramos aplicar Target Encoding. En caso de que no exista la clave, esta variable será una lista vacía
        col_encode = self.diccionario_encoding.get("target", [])

        # si hay contenido en la lista 
        if col_encode:

            target_encoder = TargetEncoder(cols=col_encode)
            df_target = target_encoder.fit_transform(df_sin_vr, self.dataframe[self.variable_respuesta])
            self.dataframe = pd.concat([self.dataframe[self.variable_respuesta].reset_index(drop=True), df_target], axis=1)

        return self.dataframe, target_encoder

    def frequency_encoding(self):
        """
        Realiza codificación de frecuencia en las columnas especificadas en el diccionario de codificación.

        Returns:
        - dataframe: DataFrame de pandas, el DataFrame con codificación de frecuencia aplicada.
        """

        # accedemos a la clave de 'frequency' para poder extraer las columnas a las que que queramos aplicar Frequency Encoding. En caso de que no exista la clave, esta variable será una lista vacía
        col_encode = self.diccionario_encoding.get("frequency", [])
        frecuencia_dict = {}  # para guardar las frecuencias calculadas
        # si hay contenido en la lista 
        if col_encode:

            # iteramos por cada una de las columnas a las que les queremos aplicar este tipo de encoding
            for categoria in col_encode:

                # calculamos las frecuencias de cada una de las categorías
                frecuencia = self.dataframe[categoria].value_counts(normalize=True)

                # guardamos las frecuencias en el diccionario
                frecuencia_dict[categoria] = frecuencia
                # mapeamos los valores obtenidos en el paso anterior, sobreescribiendo la columna original
                self.dataframe[categoria] = self.dataframe[categoria].map(frecuencia)
        
        return self.dataframe, frecuencia_dict
       

#PLOT DE OUTLIERS (ESTANDARIZACION)

def visualizar_outliers_box(df, columnas_num):
    """
    Visualiza los outliers en un conjunto de columnas numéricas de un DataFrame utilizando diagramas de caja (boxplots).

    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame que contiene las columnas numéricas a analizar.
    columnas_num : list
        Lista de nombres de las columnas numéricas que se desean visualizar.

    Retorno:
    --------
    None
        La función genera una figura con subgráficos (boxplots) para cada columna numérica de la lista. 
        Cada gráfico muestra la distribución y posibles outliers de la columna correspondiente.

    Notas:
    ------
    - La figura tiene un máximo de 64 subgráficos (8 filas x 8 columnas).
    - Si la lista de columnas excede las 64 columnas, algunos gráficos no serán representados.
    - El tamaño del gráfico total es ajustado automáticamente con `plt.tight_layout()` para evitar superposiciones.
    """
    fig , axes = plt.subplots(nrows=8, ncols=8, figsize = (15, 20) )
    axes=axes.flat
    for index,col in enumerate(columnas_num,start=0):

        sns.boxplot(x = col, data = df, ax = axes[index])
        
        
    plt.tight_layout()


def apply_frequency_encoding(df, frecuencia_dicc):
    """
    Aplica codificación por frecuencia a las columnas de un DataFrame utilizando un diccionario de frecuencias.

    Parámetros:
    -----------
    df : pd.DataFrame
        El DataFrame de entrada con las columnas categóricas a codificar.
    
    frecuencia_dicc : dict
        Un diccionario donde las claves son nombres de columnas y los valores son diccionarios 
        con las categorías y su respectiva frecuencia codificada.
        Ejemplo: 
        {
            'columna1': {'A': 0.2, 'B': 0.5, 'C': 0.3},
            'columna2': {'X': 0.4, 'Y': 0.6}
        }

    Retorna:
    --------
    pd.DataFrame
        Una copia del DataFrame original con las columnas especificadas codificadas por frecuencia.
        Si una categoría no está en el diccionario, se le asigna la mediana de los valores existentes.

    Notas:
    ------
    - Solo se codifican las columnas que están en `frecuencia_dicc`.
    - Se utiliza la función `.map()` para reemplazar las categorías por su valor de frecuencia.
    - Si una categoría no tiene un valor en el diccionario, se le asigna la mediana de los valores conocidos.

    Ejemplo:
    --------
    >>> df = pd.DataFrame({'columna1': ['A', 'B', 'C', 'D'], 'columna2': ['X', 'Y', 'X', 'Z']})
    >>> frecuencia_dicc = {'columna1': {'A': 0.2, 'B': 0.5, 'C': 0.3}, 'columna2': {'X': 0.4, 'Y': 0.6}}
    >>> apply_frequency_encoding(df, frecuencia_dicc)
       columna1  columna2
    0       0.2      0.4
    1       0.5      0.6
    2       0.3      0.4
    3       0.3      0.5  # 'D' y 'Z' no estaban en el diccionario, se les asigna la mediana

    """
    df_encoded = df.copy()
    
    for col in df.columns:
        if col in frecuencia_dicc:  # Solo encodeamos las columnas que están en el diccionario
            
            # Calculamos la mediana de la columna en el diccionario de encoding
            valores = list(frecuencia_dicc[col].to_dict().values())

            mediana_val = np.median(valores)

            # Aplicamos el encoding, usando .map() y reemplazando NaN con la mediana
            df_encoded[col] = df_encoded[col].map(frecuencia_dicc[col]).fillna(mediana_val)
    
    return df_encoded

## OUTLIERS

def identificar_outliers_iqr(df, k=1.5):
    """
    Identifica outliers en un DataFrame utilizando el método del rango intercuartílico (IQR).

    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame que contiene los datos para identificar outliers. Solo se analizarán las columnas numéricas.
    k : float, opcional
        Factor multiplicador para determinar los límites superior e inferior. 
        El valor predeterminado es 1.5, que es el estándar común para identificar outliers.

    Retorno:
    --------
    dicc_outliers : dict
        Diccionario donde las claves son los nombres de las columnas con outliers y los valores son 
        DataFrames que contienen las filas correspondientes a los outliers de esa columna.

    Efectos secundarios:
    ---------------------
    - Imprime la cantidad de outliers detectados en cada columna numérica.
    
    Notas:
    ------
    - El método utiliza el rango intercuartílico para calcular los límites:
        * Límite superior = Q3 + (IQR * k)
        * Límite inferior = Q1 - (IQR * k)
      Donde Q1 y Q3 son los percentiles 25 y 75, respectivamente.
    - Las columnas no numéricas son ignoradas.
    - Si una columna no contiene outliers, no se incluirá en el diccionario de retorno.
    - Los valores NaN en las columnas no son considerados para el cálculo del IQR.

    Ejemplo:
    --------
    >>> df = pd.DataFrame({'A': [1, 2, 3, 100], 'B': [5, 6, 7, 8]})
    >>> outliers = identificar_outliers_iqr(df)
    La columna A tiene 1 outliers
    La columna B tiene 0 outliers
    >>> print(outliers)
    {'A':     A  B
           3  100  8}
    """
    df_num=df.select_dtypes(include=np.number)
    dicc_outliers={}
    for columna in df_num.columns:
        Q1, Q3= np.nanpercentile(df[columna], (25, 75))
        iqr= Q3 - Q1
        limite_superior= Q3 + (iqr * k)
        limite_inferior= Q1 - (iqr * k)

        condicion_sup= df[columna] > limite_superior
        condicion_inf= df[columna] < limite_inferior
        df_outliers= df[condicion_inf | condicion_sup]
        print(f"La columna {columna.upper()} tiene {df_outliers.shape[0]} outliers")
        if not df_outliers.empty:
            dicc_outliers[columna]= df_outliers

    return dicc_outliers    


#DESBALANCES

class Desbalanceo:
    """
    Clase para manejar problemas de desbalanceo en conjuntos de datos.

    Atributos:
    ----------
    dataframe : pandas.DataFrame
        Conjunto de datos con las clases desbalanceadas.
    variable_dependiente : str
        Nombre de la variable objetivo que contiene las clases a balancear.

    Métodos:
    --------
    visualizar_clase(color="orange", edgecolor="black"):
        Genera un gráfico de barras para visualizar la distribución de clases.
    balancear_clases_pandas(metodo):
        Balancea las clases utilizando técnicas de sobremuestreo o submuestreo con pandas.
    balancear_clases_imblearn(metodo):
        Balancea las clases utilizando RandomOverSampler o RandomUnderSampler de imbalanced-learn.
    balancear_clases_smote():
        Aplica el método SMOTE para generar nuevas muestras de la clase minoritaria.
    balancear_clase_smotenc(columnas_categoricas__encoded, sampling_strategy="auto"):
        Aplica SMOTENC para balancear clases en datos que contienen columnas categóricas codificadas.
    balancear_clases_tomek(sampling_strategy="auto"):
        Aplica el método Tomek Links para eliminar pares cercanos entre clases.
    balancear_clases_smote_tomek():
        Aplica SMOTE combinado con Tomek Links para balancear las clases.
    """
    def __init__(self, dataframe, variable_dependiente):
        self.dataframe = dataframe
        self.variable_dependiente = variable_dependiente

    def visualizar_clase(self, color="orange", edgecolor="black"):
        """
        Visualiza la distribución de clases de la variable dependiente.

        Parámetros:
        -----------
        color : str, opcional
            Color de las barras del gráfico. Por defecto, "orange".
        edgecolor : str, opcional
            Color del borde de las barras. Por defecto, "black".

        Retorna:
        --------
        None
            Muestra un gráfico de barras que representa la distribución de clases de la variable dependiente.

        Ejemplo:
        --------
        >>> desbalanceo = Desbalanceo(dataframe, "target")
        >>> desbalanceo.visualizar_clase()
        """
        plt.figure(figsize=(8, 5))  # para cambiar el tamaño de la figura
        fig = sns.countplot(data=self.dataframe, 
                            x=self.variable_dependiente,  
                            color=color,  
                            edgecolor=edgecolor)
        fig.set(xticklabels=["No", "Yes"])
        plt.show()

    def balancear_clases_pandas(self, metodo):
        """
        Balancea las clases utilizando técnicas de sobremuestreo o submuestreo con pandas.

        Parámetros:
        -----------
        metodo : str
            Método de balanceo a utilizar. Puede ser:
            - "downsampling": Reduce el número de muestras de la clase mayoritaria.
            - "upsampling": Incrementa el número de muestras de la clase minoritaria.

        Retorna:
        --------
        pd.DataFrame
            DataFrame con las clases balanceadas.

        Lanza:
        -------
        ValueError
            Si el método proporcionado no es "downsampling" o "upsampling".

        Ejemplo:
        --------
        >>> desbalanceo = Desbalanceo(dataframe, "target")
        >>> df_balanceado = desbalanceo.balancear_clases_pandas("downsampling")
        """

        # Contar las muestras por clase
        contar_clases = self.dataframe[self.variable_dependiente].value_counts()
        clase_mayoritaria = contar_clases.idxmax()
        clase_minoritaria = contar_clases.idxmin()

        # Separar las clases
        df_mayoritaria = self.dataframe[self.dataframe[self.variable_dependiente] == clase_mayoritaria]
        df_minoritaria = self.dataframe[self.dataframe[self.variable_dependiente] == clase_minoritaria]

        if metodo == "downsampling":
            # Submuestrear la clase mayoritaria
            df_majority_downsampled = df_mayoritaria.sample(contar_clases[clase_minoritaria], random_state=42)
            # Combinar los subconjuntos
            df_balanced = pd.concat([df_majority_downsampled, df_minoritaria])

        elif metodo == "upsampling":
            # Sobremuestrear la clase minoritaria
            df_minority_upsampled = df_minoritaria.sample(contar_clases[clase_mayoritaria], replace=True, random_state=42)
            # Combinar los subconjuntos
            df_balanced = pd.concat([df_mayoritaria, df_minority_upsampled])

        else:
            raise ValueError("Método no reconocido. Use 'downsampling' o 'upsampling'.")

        return df_balanced

    def balancear_clases_imblearn(self, metodo):
        """
        Balancea las clases utilizando las técnicas RandomOverSampler o RandomUnderSampler 
        de la biblioteca imbalanced-learn.

        Parámetros:
        -----------
        metodo : str
            Método de balanceo a utilizar. Puede ser:
            - "RandomOverSampler": Sobremuestreo aleatorio de la clase minoritaria.
            - "RandomUnderSampler": Submuestreo aleatorio de la clase mayoritaria.

        Retorna:
        --------
        pd.DataFrame
            DataFrame con las clases balanceadas.

        Lanza:
        -------
        ValueError
            Si el método proporcionado no es "RandomOverSampler" o "RandomUnderSampler".

        Ejemplo:
        --------
        >>> desbalanceo = Desbalanceo(dataframe, "target")
        >>> df_balanceado = desbalanceo.balancear_clases_imblearn("RandomOverSampler")
        """

        X = self.dataframe.drop(columns=[self.variable_dependiente])
        y = self.dataframe[self.variable_dependiente]

        if metodo == "RandomOverSampler":
            ros = RandomOverSampler(random_state=42)
            X_resampled, y_resampled = ros.fit_resample(X, y)

        elif metodo == "RandomUnderSampler":
            rus = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = rus.fit_resample(X, y)

        else:
            raise ValueError("Método no reconocido. Use 'RandomOverSampler' o 'RandomUnderSampler'.")

        df_resampled = pd.concat([pd.DataFrame(X_resampled), pd.Series(y_resampled, name=self.variable_dependiente)], axis=1)
        return df_resampled
    
    def balancear_clases_smote(self):
        """
        Aplica el método SMOTE (Synthetic Minority Oversampling Technique) para balancear clases 
        generando nuevas muestras sintéticas para la clase minoritaria.

        Returns:
        --------
        pd.DataFrame
            DataFrame balanceado con las nuevas muestras generadas por SMOTE.

        Notas:
        ------
        - Este método es útil cuando la clase minoritaria tiene muy pocas muestras.
        - SMOTE genera muestras sintéticas interpolando entre los puntos de la clase minoritaria.
        """
        X = self.dataframe.drop(columns=[self.variable_dependiente])
        y = self.dataframe[self.variable_dependiente]

        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=self.variable_dependiente)], axis=1)
        return df_resampled

    def balancear_clase_smotenc(self, columnas_categoricas__encoded,sampling_strategy="auto"):
        """
        Balancea las clases utilizando el método SMOTENC, diseñado para trabajar con variables categóricas.

        Parámetros:
        -----------
        columnas_categoricas_encoded : list
            Lista de índices que indican cuáles columnas son categóricas (previamente codificadas).
        sampling_strategy : str o float, opcional
            Estrategia de muestreo, por defecto "auto".

        Retorna:
        --------
        pd.DataFrame
            DataFrame con las clases balanceadas usando SMOTENC.

        Ejemplo:
        --------
        >>> desbalanceo = Desbalanceo(dataframe, "target")
        >>> df_balanceado = desbalanceo.balancear_clase_smotenc([0, 1], sampling_strategy="minority")
        """
        X = self.dataframe.drop(columns=[self.variable_dependiente])
        y = self.dataframe[self.variable_dependiente]

        smotenc = SMOTENC(random_state=42, categorical_features=columnas_categoricas__encoded,sampling_strategy=sampling_strategy)
        X_resampled, y_resampled = smotenc.fit_resample(X, y)
        
        df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=self.variable_dependiente)], axis=1)
        return df_resampled, smotenc

    def balancear_clases_tomek(self,sampling_strategy="auto"):
        """
        Aplica el método de Tomek Links para balancear clases eliminando pares cercanos
        entre la clase mayoritaria y la minoritaria.
        
        Returns:
            pd.DataFrame: DataFrame balanceado tras aplicar Tomek Links.
        """
        X = self.dataframe.drop(columns=[self.variable_dependiente])
        y = self.dataframe[self.variable_dependiente]

        tomek = TomekLinks(sampling_strategy=sampling_strategy)
        X_resampled, y_resampled = tomek.fit_resample(X, y)
        df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), 
                                pd.Series(y_resampled, name=self.variable_dependiente)], axis=1)

        return df_resampled, tomek


    def balancear_clases_smote_tomek(self):
        """
        Balancea las clases utilizando una combinación de SMOTE y Tomek Links para manejar el desbalanceo.

        Retorna:
        --------
        pd.DataFrame
            DataFrame con las clases balanceadas utilizando SMOTETomek.

        Ejemplo:
        --------
        >>> desbalanceo = Desbalanceo(dataframe, "target")
        >>> df_balanceado = desbalanceo.balancear_clases_smote_tomek()
        """
        X = self.dataframe.drop(columns=[self.variable_dependiente])
        y = self.dataframe[self.variable_dependiente]

        smote_tomek = SMOTETomek(random_state=42)
        X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
        
        df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=self.variable_dependiente)], axis=1)
        return df_resampled
    

        

def plot_matriz_confusion(matriz_confusion, invertir=True, tamano_grafica=(4, 3), labels=False, label0="", label1="", color="Purples"):
    """
    Genera un heatmap para visualizar una matriz de confusión.

    Args:
        matriz_confusion (numpy.ndarray): Matriz de confusión que se desea graficar.
        invertir (bool, opcional): Si es True, invierte el orden de las filas y columnas de la matriz
            para reflejar el eje Y invertido (orden [1, 0] en lugar de [0, 1]). Por defecto, True.
        tamano_grafica (tuple, opcional): Tamaño de la figura en pulgadas. Por defecto, (4, 3).
        labels (bool, opcional): Si es True, permite agregar etiquetas personalizadas a las clases
            utilizando `label0` y `label1`. Por defecto, False.
        label0 (str, opcional): Etiqueta personalizada para la clase 0 (negativa). Por defecto, "".
        label1 (str, opcional): Etiqueta personalizada para la clase 1 (positiva). Por defecto, "".

    Returns:
        None: La función no retorna ningún valor, pero muestra un heatmap con la matriz de confusión.

    Ejemplos:
>             from sklearn.metrics import confusion_matrix
>         >>> y_true = [0, 1, 1, 0, 1, 1]
>         >>> y_pred = [0, 1, 1, 0, 0, 1]
>         >>> matriz_confusion = confusion_matrix(y_true, y_pred)
>         >>> plot_matriz_confusion(matriz_confusion, invertir=True, labels=True, label0="Clase Negativa", label1="Clase Positiva")
    """
    if invertir == True:
        plt.figure(figsize=(tamano_grafica))
        if labels == True:
            labels = [f'1: {label1}', f'0: {label0}']
        else:
            labels = [f'1', f'0']
        sns.heatmap(matriz_confusion[::-1, ::-1], annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap= color)
        plt.xlabel("Predicción")
        plt.ylabel("Real");
    else: 
        plt.figure(figsize=(tamano_grafica))
        if labels == True:
            labels = [f'0: {label0}', f'1: {label1}']
        else:
            labels = [f'0', f'1']
        sns.heatmap(matriz_confusion, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap= color)    
        plt.xlabel("Predicción")
        plt.ylabel("Real");  

