# Entrega final: Aumento de valor en la vivienda por la instalación de Fotovoltaica

   
![Energía Fotovoltaica en Viviendas](https://github.com/jgilsu11/Proyecto_Final_Fotovoltaica/blob/main/Multimedia/Imagen_proyecto_final_readme.webp)

---

## Descripción

Este proyecto consiste en la especificación, preprocesamiento y entrenamiento de modelos predictivos para determinar la probabilidad de que una vivienda tenga energía fotovoltaica debido a la falta de acceso a esos datos y posteriormente su impacto en el valor de la vivienda. Se han utilizado diversas técnicas de análisis y modelización para lograr resultados óptimos.

Las técnicas utilizadas incluyen:
- Preprocesamiento de datos (gestión de nulos, eliminación de duplicados, encoding, estandarización, tratamiento de desbalanceo y detección de outliers).
- Generación y entrenamiento de modelos predictivos.
- Análisis exploratorio de datos (EDA) y segmentación de viviendas.
- Creación de un dashboard interactivo para visualizar los resultados con Power BI.
- Almacenamiento de datos en MongoDB.

Adicionalmente, se ha utilizado información obtenida mediante scraping para mejorar la calidad de los datos.

---

## Estructura del Proyecto

El desarrollo del proyecto se ha organizado de la siguiente manera:

```
├── Dashboard/                              # Contiene el dashboard y la presentación del proyecto
│   ├── Dashboard_Fotovoltaica.pbix         # Dashboard interactivo en Power BI
│   ├── Presentacion_Google.pptx            # Presentación del proyecto
│
├── datos/                                  # Datos utilizados y generados en el proyecto
│   ├── EF/                                 # Datos de energía fotovoltaica (para predicción)
│   │   ├── Datos_alimentados/              # Datos tras primera carga ya alimentados
│   │   ├── Datos_dashboard/                # Datos listos para visualización en el dashboard
│   │   ├── Datos_filtrados/                # Datos tras aplicar filtros relevantes
│   │   ├── Datos_modelos/                  # Datos utilizados para entrenar modelos
│   │   ├── Datos_segmentados/              # Datos segmentados según tamaño de vivienda
│   ├── red_piso/                           # Datos obtenidos de Red Piso
│   │   ├── Datos_extraccion/               # Datos brutos extraídos mediante web scraping
│   │   ├── Datos_formateados/              # Datos tras limpieza y reestructuración
│   │   ├── Datos_preprocesados/            # Datos listos para aplicar los modelos
│   ├── ymls/                               # Archivos YML
│   ├── EPCOVivienda_2021.csv               # Archivo CSV base del análisis
│   ├── mapeos.xlsx                         # Archivo con mapeos
│
├── Notebooks/                               # Notebooks de Jupyter para análisis y modelado
│   ├── 3.4_Modelos_optimizacion.ipynb       # Optimización de hiperparámetros en modelos
│   ├── Modelos_amplias/                     # Modelos de las viviendas amplias
│   │   ├── amplias_optimizacion.ipynb       # Optimización de las viviendas amplias
│   │   ├── preprocesamiento_amplias.ipynb   # Preprocesamiento de las viviendas amplias
│   ├── Modelos_grandes/                     # Modelos de las viviendas grandes
│   │   ├── grandes_optimizacion.ipynb       # Optimización de las viviendas grandes
│   │   ├── preprocesamiento_grandes.ipynb   # Preprocesamiento de las viviendas grandes
│   ├── Modelos_medianas/                    # Modelos de las viviendas medianas
│   │   ├── medianas_optimizacion.ipynb      # Optimización de las viviendas medianas
│   │   ├── preprocesamiento_medianas.ipynb  # Preprocesamiento de las viviendas medianas
│   ├── Modelos_pequeñas/                    # Modelos de las viviendas pequeñas
│   │   ├── pequeñas_optimizacion.ipynb      # Optimización de las viviendas pequeñas
│   │   ├── preprocesamiento_pequeñas.ipynb  # Preprocesamiento de las viviendas pequeñas
│   ├── 1_Extraccion_FV.ipynb                # Extracción de datos para predecir
│   ├── 2_carga_y_EDA.ipynb                  # Carga y análisis exploratorio de datos para predecir
│   ├── 3_1_segmentacion.ipynb               # Segmentación por tamaño de los datos para predecir
│   ├── 3_2_EDA_y_Outliers.ipynb             # Análisis exploratorio de datos para predecir y tratamiento de Outliers
│   ├── 3_3_Modelos_iniciales.ipynb          # Primer acercamiento a los modelos predictivos usando automl
│   ├── 4_extraccion_pisos.ipynb             # Extracción de los datos de Red piso donde aplicar los modelos
│   ├── 5_EDA_y_carga_pisos.ipynb            # Carga y análisis exploratorio de datos de Red piso
│   ├── 6_1_segmentacion_pisos.ipynb         # Segmentación por tamaño de los datos de Red piso donde aplicar los modelos
│   ├── 6_2_EDA_y_Outliers.ipynb             # Análisis exploratorio de datos de Red piso y tratamiento de Outliers
│   ├── 6_3_preprocesamiento_pisos.ipynb     # Preprocesamiento necesario de los datos de Red piso para aplicar los modelos
│   ├── 6_4_prediccion_pisos.ipynb           # Predicción usando los datos de Red piso
│
├── src/                                     # Scripts en Python
│   ├── soporte_ajuste_modelos.py            # Ajuste de modelos
│   ├── soporte_carga.py                     # Funciones de carga de datos
│   ├── soporte_extraccion.py                # Scraping 
│   ├── soporte_outliers.py                  # Detección y tratamiento de outliers
│   ├── soporte_prediccion.py                # Predicción con modelos
│   ├── soporte_preprocesamiento.py          # Preprocesamiento general
│
├── transformers/                            # Modelos y preprocesamientos guardados
│   ├── modelos/                             # Pickles de modelos entrenados (Están el enlace de Drive que se encuentra más abajo)
│   ├── preprocesamiento/                    # Pickles de preprocesamiento
│
├── .gitignore                               # Archivos a ignorar en el repositorio
├── README.md                                # Documentación del proyecto actual
```

Aquí está el enlace para los modelos predictivos más pesados: https://drive.google.com/drive/folders/1WJGWWn3AxBDvA75PuM4CAVRuieE7R3gm?usp=sharing  

---

## Requisitos e Instalación 🛠️

Este proyecto usa Python 3.11.9 y bibliotecas como:

- pandas, numpy, matplotlib, seaborn
- scikit-learn, shap, pickle, tqdm
- requests, sys, os

Se recomienda crear dos entornos virtuales uno general y otro únicamente para lo relacionado con automel (Pycaret) para asegurar la compatibilidad de bibliotecas.

---

## Tabla Resumen

| **Modelo**    | **Algoritmo**       | **Recall (Test)**   | **Overfitting (Diff Recall)**    | **Kappa (Test)** | **AUC (Test)** | **F1 (Test)** |
|---------------|---------------------|---------------------|----------------------------------|--------------------|----------------|---------------|
| **Modelo Grandes** | Random Forest     | **0.97**          | **0.02**                  | **0.93**     | **0.99**   |**0.97**     |
| **Modelo Amplias** | Random Forest     | **0.98**          | **0.01**                  | **0.97**     | **1.00**   |**0.98**     |
| **Modelo Medianas** | Random Forest    | **0.98**          | **0.01**                  | **0.96**     | **1.00**   |**0.98**     |
| **Modelo Pequeñas** | Random Forest    | **0.99**          | **0.01**                  | **0.97**     | **1.00**   |**0.99**     |


---

## Aportación 🤝

Este proyecto permite conocer detalladamente el perfil de las viviendas con energía fotovoltaica y su impacto en el precio de la vivienda. Esto puede ser útil para focalizar los esfuerzos de capatación de clientes por parte de una empresa del sector a los perfiles con más probabilidad de adoptar dicha tecnología, así cómo permite al cliente conocer de manera sencilla y clara los beneficios de su instalación.

---

## Próximos Pasos 🚀

1. Desarrollar una aplicación en Streamlit para que los usuarios puedan verificar si su vivienda cumple el perfil de energía fotovoltaica.
2. Conseguir más datos adicionales a los de Red Piso para mejorar la representatividad de viviendas con fotovoltaica.
3. Ajustar los resultados reflejados en el dashboard para reducir sesgos en el análisis de variaciones de precio (Unido al punto 2).
