import pandas as pd

#para la extracción de datos
import requests  # Requests permite hacer peticiones HTTP para interactuar con APIs y sitios web.
from bs4 import BeautifulSoup  # BeautifulSoup permite parsear HTML y extraer información del contenido de las páginas web.
from time import sleep







def sacar_tabla(url):
    """
    Extrae las tablas relevantes de una página web, sus encabezados y el contenido, y devuelve un DataFrame consolidado.

    Args:
        url (str): URL de la página web de la cual se desea extraer las tablas.

    Returns:
        tuple: Una tupla que contiene:
            - encabezados (list): Una lista con los encabezados de las tablas extraídas.
            - df_radiacion (pd.DataFrame): Un DataFrame que consolida el contenido de las tablas seleccionadas.

    Funcionalidad:
        1. Realiza una solicitud HTTP GET a la URL proporcionada y obtiene su contenido HTML.
        2. Utiliza BeautifulSoup para analizar el HTML y encuentra todas las tablas en la página.
        3. Itera sobre las tablas (excepto la primera) para:
            - Extraer los títulos (`caption`) y agregarlos a una lista.
            - Obtener los encabezados (`th`) de las tablas.
            - Extraer el contenido de las filas (`tr`), procesándolo para limpiarlo y convertirlo en un formato estructurado.
        4. Combina el contenido de todas las tablas seleccionadas en un único DataFrame.

    Notas:
        - La función ignora la primera tabla de la página.
        - Los datos de las filas son limpiados para manejar puntos y comas en números.
        - El DataFrame final consolida los contenidos de las tablas relevantes.

    Ejemplo:
        ```python
        url = "https://ejemplo.com/datos"
        encabezados, df = sacar_tabla(url)
        print(encabezados)
        print(df.head())
        ```
    """
    res= requests.get(url)
    sopa_radiacion= BeautifulSoup(res.content, "html.parser")
    tablas = sopa_radiacion.findAll("table")

    #me quedo solo con las tablas que me interesan
    lista_titulos=[]
    df_radiacion=pd.DataFrame()
    for k in range(1,len(tablas)):
        #consigo los nombres de las tablas
        lista_titulos = tablas[k].findAll("caption")
        titulos=[nombre.getText() for nombre in lista_titulos]
        lista_titulos.append(titulos)

        #saco los encabezados
        lista_encabezados = tablas[k].findAll("th")
        encabezados=[nombre.getText() for nombre in lista_encabezados]

        #saco el contenido de la tabla

        lista_contenido= tablas[k].findAll("tr")
        resultados=[]
        contenidos=[nombre.getText() for nombre in lista_contenido]
        for i in range(1,len(contenidos)):

            elementos_fila=contenidos[i].replace(".","").replace(",",".").split("\n")[1:-1]

            #añado la lista de elementos una la lista
            resultados.append(elementos_fila)
        df=pd.DataFrame(resultados)
        df_radiacion=pd.concat([df_radiacion,df],axis=0, ignore_index=True)
    return encabezados, df_radiacion

