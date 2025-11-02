"""
Módulo de preparación de datos para el proyecto Turkish Music Emotion.

Este script realiza:
1. Carga de datos raw desde CSV
2. Limpieza estructural (columnas, valores, tipos)
3. Filtrado de clases válidas
4. División estratificada train/test
5. Guardado de datasets procesados

Uso:
    python -m eq18_turkish_music_mlops.prepare
"""

import pandas as pd
import logging
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split

from eq18_turkish_music_mlops.utils.logger import setup_logging
import sys

# Configuración de loggin
setup_logging()

logger = logging.getLogger(__name__)


def load_params(params_path="params.yaml"):
    """
    Carga parámetros desde archivo YAML.
    
    Args:
        params_path (str): Ruta al archivo de parámetros
        
    Returns:
        dict: Diccionario con parámetros
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        yaml.YAMLError: Si hay errores de sintaxis en YAML
    """
    try:
        with open(params_path, "r", encoding="utf-8") as f:
            params = yaml.safe_load(f)
        logger.info(f"Parámetros cargados desde {params_path}")
        return params
    except FileNotFoundError:
        logger.error(f"Archivo {params_path} no encontrado")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error al parsear YAML: {e}")
        raise


def validate_dataframe(df, target_col, required_cols=None):
    """
    Valida estructura básica del dataframe.
    
    Args:
        df (pd.DataFrame): DataFrame a validar
        target_col (str): Nombre de la columna objetivo
        required_cols (list, optional): Columnas requeridas
        
    Raises:
        ValueError: Si faltan columnas críticas
    """
    if df.empty:
        raise ValueError("DataFrame está vacío")
    
    if target_col not in df.columns:
        raise ValueError(f"Columna objetivo '{target_col}' no encontrada")
    
    if required_cols:
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Columnas faltantes: {missing}")
    
    logger.info(f"Validación exitosa. Shape: {df.shape}")


def clean_column_names(df):
    """
    Limpia nombres de columnas (espacios, caracteres especiales).
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame con columnas limpias
    """
    original_cols = df.columns.tolist()
    df.columns = df.columns.str.strip()
    
    renamed = [col for orig, col in zip(original_cols, df.columns) if orig != col]
    if renamed:
        logger.info(f"Columnas renombradas: {len(renamed)}")
    
    return df


def remove_irrelevant_columns(df, cols_to_drop=None):
    """
    Elimina columnas irrelevantes identificadas en EDA.
    
    Args:
        df (pd.DataFrame): DataFrame original
        cols_to_drop (list): Lista de columnas a eliminar
        
    Returns:
        pd.DataFrame: DataFrame sin columnas irrelevantes
    """
    if cols_to_drop is None:
        cols_to_drop = ["mixed_type_col"]
    
    existing_drops = [col for col in cols_to_drop if col in df.columns]
    
    if existing_drops:
        df = df.drop(columns=existing_drops)
        logger.info(f"Columnas eliminadas: {existing_drops}")
    else:
        logger.info("No se encontraron columnas irrelevantes para eliminar")
    
    return df


def clean_target_column(df, target_col, valid_classes):
    """
    Limpia y filtra la columna objetivo.
    
    Args:
        df (pd.DataFrame): DataFrame original
        target_col (str): Nombre de columna objetivo
        valid_classes (list): Lista de clases válidas
        
    Returns:
        pd.DataFrame: DataFrame con target limpio
    """
    initial_count = len(df)
    
    # Normalizar valores (espacios, minúsculas)
    df[target_col] = df[target_col].str.strip().str.lower()
    
    # Filtrar clases válidas
    df = df[df[target_col].isin(valid_classes)]
    
    filtered_count = initial_count - len(df)
    if filtered_count > 0:
        logger.warning(f"Filas filtradas por clases inválidas: {filtered_count}")
    
    # Verificar distribución
    class_dist = df[target_col].value_counts()
    logger.info(f"Distribución de clases:\n{class_dist}")
    
    return df


def convert_numeric_columns(df, target_col):
    """
    Convierte columnas con formato de texto a numérico.
    
    Procesa columnas tipo object (excepto target) que contienen números
    con comas decimales o espacios. Convierte a float64.
    
    Args:
        df (pd.DataFrame): DataFrame original
        target_col (str): Nombre de columna objetivo (no procesar)
        
    Returns:
        pd.DataFrame: DataFrame con columnas numéricas convertidas
    """
    object_cols = df.select_dtypes(include="object").columns.tolist()
    numeric_candidates = [col for col in object_cols if col != target_col]
    
    converted_count = 0
    
    for col in numeric_candidates:
        try:
            # Limpiar formato: comas -> puntos, eliminar espacios
            df[col] = (
                df[col].astype(str)
                .str.replace(",", ".", regex=False)
                .str.replace(" ", "", regex=False)
                .str.strip()
            )
            
            # Convertir a numérico
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
            # Redondear a 3 decimales para consistencia
            df[col] = df[col].round(3)
            
            converted_count += 1
            
        except Exception as e:
            logger.warning(f"No se pudo convertir columna '{col}': {e}")
    
    logger.info(f"Columnas numéricas convertidas: {converted_count}")
    
    return df


def remove_duplicates(df):
    """
    Elimina filas duplicadas del dataset.
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame sin duplicados
    """
    initial_count = len(df)
    df = df.drop_duplicates()
    duplicates_removed = initial_count - len(df)
    
    if duplicates_removed > 0:
        logger.warning(f"Duplicados eliminados: {duplicates_removed}")
    else:
        logger.info("No se encontraron duplicados")
    
    return df


def check_missing_values(df):
    """
    Reporta valores faltantes en el dataset.
    
    Args:
        df (pd.DataFrame): DataFrame a inspeccionar
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Missing': missing,
        'Percentage': missing_pct
    })
    missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False)
    
    if not missing_df.empty:
        logger.warning(f"Valores faltantes encontrados:\n{missing_df}")
    else:
        logger.info("No hay valores faltantes")


def split_train_test(df, target_col, test_size, random_state):
    """
    Divide dataset en train y test con estratificación.
    
    Args:
        df (pd.DataFrame): DataFrame completo
        target_col (str): Columna objetivo para estratificar
        test_size (float): Proporción de test (0.0 - 1.0)
        random_state (int): Semilla para reproducibilidad
        
    Returns:
        tuple: (train_df, test_df)
        
    Raises:
        ValueError: Si test_size no está en rango válido
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size debe estar entre 0 y 1, recibido: {test_size}")
    
    logger.info(f"Dividiendo datos: test_size={test_size}, random_state={random_state}")
    
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col]
    )
    
    # Logs de distribución
    logger.info(f"Train set: {len(train)} samples ({len(train)/len(df)*100:.1f}%)")
    logger.info(f"Test set: {len(test)} samples ({len(test)/len(df)*100:.1f}%)")
    
    logger.info(f"Distribución train:\n{train[target_col].value_counts()}")
    logger.info(f"Distribución test:\n{test[target_col].value_counts()}")
    
    return train, test


def save_datasets(train, test, processed_dir):
    """
    Guarda datasets procesados en CSV.
    
    Args:
        train (pd.DataFrame): Dataset de entrenamiento
        test (pd.DataFrame): Dataset de prueba
        processed_dir (Path): Directorio de salida
    """
    processed_dir.mkdir(exist_ok=True, parents=True)
    
    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"
    
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    
    logger.info(f"Datasets guardados en {processed_dir}/")
    logger.info(f"  - train.csv: {train.shape}")
    logger.info(f"  - test.csv: {test.shape}")


def main():
    """
    Función principal del pipeline de preparación.
    
    Orquesta todo el proceso de preparación de datos siguiendo
    el flujo: carga -> limpieza -> transformación -> división -> guardado
    """
    try:
        # Crear directorio de logs
        Path("logs").mkdir(exist_ok=True)
        
        logger.info("="*60)
        logger.info("INICIANDO PIPELINE DE PREPARACIÓN DE DATOS")
        logger.info("="*60)
        
        # 1. Cargar parámetros
        params = load_params()
        
        raw_path = params["paths"]["raw_data"]
        processed_dir = Path(params["paths"]["processed_dir"])
        test_size = params["data"]["test_size"]
        random_state = params["data"]["random_state"]
        target_col = params["data"]["target_col"]
        valid_classes = params["data"]["valid_classes"]
        
        # 2. Cargar datos
        logger.info(f"Cargando datos desde: {raw_path}")
        df = pd.read_csv(raw_path)
        logger.info(f"Dataset cargado: {df.shape} (filas x columnas)")
        
        # 3. Validación inicial
        validate_dataframe(df, target_col)
        
        # 4. Limpieza estructural
        df = clean_column_names(df)
        df = remove_irrelevant_columns(df)
        
        # 5. Limpieza de target
        df = clean_target_column(df, target_col, valid_classes)
        
        # 6. Conversión de tipos
        df = convert_numeric_columns(df, target_col)
        
        # 7. Eliminar duplicados
        df = remove_duplicates(df)
        
        # 8. Reportar valores faltantes
        check_missing_values(df)
        
        # 9. División train/test
        train, test = split_train_test(df, target_col, test_size, random_state)
        
        # 10. Guardar datasets
        save_datasets(train, test, processed_dir)
        
        logger.info("="*60)
        logger.info("PREPARACIÓN COMPLETADA EXITOSAMENTE")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Error crítico en prepare.py: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()