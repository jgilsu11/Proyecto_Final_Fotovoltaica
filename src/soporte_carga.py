import pandas as pd
import numpy as np

import pymongo
from pymongo import MongoClient

from pymongo import MongoClient
from pymongo.database import Database
from pymongo.server_api import ServerApi


import dotenv
from dotenv import load_dotenv
load_dotenv()

import sys 
import os




def probar_con_atlas():   #Meter la url a .env
    """
    Prueba la conexión con una base de datos MongoDB Atlas utilizando la URI almacenada en variables de entorno.

    La función recupera la URI desde la variable de entorno `"Uri_atlas"` y establece una conexión con el servidor 
    de MongoDB Atlas. Luego, envía un comando `ping` para verificar si la conexión es exitosa.

    Parámetros:
    -----------
    No recibe parámetros.

    Retorna:
    --------
    None
        Imprime un mensaje en la consola indicando si la conexión fue exitosa o si ocurrió un error.

    Notas:
    ------
    - La URI debe estar almacenada en una variable de entorno (`.env`) con el nombre `"Uri_atlas"`.
    - En caso de error en la conexión, se captura la excepción y se imprime el mensaje de error.

    Ejemplo:
    --------
    >>> probar_conn_atlas()
    Pinged your deployment. You successfully connected to MongoDB!
    """
    uri = os.getenv("Uri_atlas")
    # Create a new client and connect to the server
    client = MongoClient(uri, server_api=ServerApi('1'))
    # Send a ping to confirm a successful connection
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)



def conectar_mongo():
    """
    Conecta a la base de datos MongoDB usando variables de entorno.
    
    Returns:
        Cliente MongoDB y base de datos conectada.
    """
    uri = os.getenv("Uri_atlas")  #para ir a atlas
    db_name = "ProyectoFinal"
    client = MongoClient(uri)
    db = client[db_name]
    return client, db


def insertar_en_coleccion(db: Database, nombre_coleccion: str, datos, silent_mode = False):
    """
    Inserta nuevos documentos en la colección especificada en MongoDB.
    
    Args:
        db: Conexión a la base de datos MongoDB.
        nombre_coleccion: Nombre de la colección donde se almacenarán los datos.
        datos: Puede ser una ruta a un archivo CSV o un DataFrame de pandas.
    """
    collection = db[nombre_coleccion]
    
    # Cargar datos desde un CSV si es una ruta
    if isinstance(datos, str):
        if not os.path.exists(datos):
            raise FileNotFoundError(f"El archivo {datos} no existe.")
        df = pd.read_csv(datos)
    elif isinstance(datos, pd.DataFrame):
        df = datos
    else:
        raise ValueError("El parámetro 'datos' debe ser una ruta a un archivo CSV o un DataFrame.")
    
    batch_size = 5000  # Número de documentos por inserción
    for i in range(0, len(df), batch_size):
        collection.insert_many(df.iloc[i:i+batch_size].to_dict(orient="records"))

    if not silent_mode:
        print(f"✅ Datos insertados o actualizados en la colección '{nombre_coleccion}' correctamente.")