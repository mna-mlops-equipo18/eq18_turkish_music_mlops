# eq18_turkish_music_mlops notes!

## Notas relacionadas a la generación de versión 3_eq18_FinalFase_1_monolitico.ipynb

La división del conjunto de datos en validación, entrenamiento y pruebas se hace en “2_Ali_Campos_initial_data_exploration.ipynb” sobre “df_no_outliers”. Sin embargo, el cálculo del Rango Intercuartílico (IQR) y, por lo tanto, los umbrales utilizados para identificar y reemplazar los outliers, se basan en los cuartiles (Q1 y Q3) de toda la distribución, incluyendo los datos de prueba, por lo que introduce data leakage.

Tipo de Preprocesamiento: Limpieza Estructural
Por lo que se propone sólo aplicar cambios a todo el conjunto de datos que sólo afecten la estructura intrínseca y que no se basen en estadísticas calculadas sobre la distribución de las variables. Como por ejemplo, la conversión de tipos de datos de objetos a numérico a las variables que aplique, estandarización de texto y typos en las variables categóricas y eliminación de duplicados. Finalmente, eliminación de columnas con tantos vacíos que no aportan al modelo según referencias bibliográficas, como la columna "mixed_type_col".

