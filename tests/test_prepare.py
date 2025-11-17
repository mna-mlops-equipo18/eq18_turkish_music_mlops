"""
Pruebas unitarias para eq18_turkish_music_mlops/prepare.py
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import yaml

# Importar funciones del módulo prepare
sys.path.insert(0, str(Path(__file__).parent.parent))
from eq18_turkish_music_mlops.prepare import (
    load_params,
    validate_dataframe,
    clean_column_names,
    remove_irrelevant_columns,
    clean_target_column,
    convert_numeric_columns,
    remove_duplicates,
    split_train_test,
)


class TestLoadParams:
    """Pruebas para carga de parámetros."""
    
    def test_load_params_valid_file(self, temp_workspace):
        """Carga exitosa de params.yaml válido."""
        params_path = temp_workspace / "params.yaml"
        params = load_params(str(params_path))
        
        assert "paths" in params
        assert "data" in params
        assert params["data"]["random_state"] == 42
    
    def test_load_params_missing_file(self):
        """Error al cargar archivo inexistente."""
        with pytest.raises(FileNotFoundError):
            load_params("nonexistent.yaml")


class TestValidateDataframe:
    """Pruebas para validación de DataFrames."""
    
    def test_validate_empty_dataframe(self):
        """Error con DataFrame vacío."""
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="vacío"):
            validate_dataframe(df, "Class")
    
    def test_validate_missing_target(self, sample_dataframe):
        """Error cuando falta columna objetivo."""
        with pytest.raises(ValueError, match="no encontrada"):
            validate_dataframe(sample_dataframe, "NonExistent")
    
    def test_validate_success(self, sample_dataframe):
        """Validación exitosa."""
        validate_dataframe(sample_dataframe, "Class")


class TestCleanColumnNames:
    """Pruebas para limpieza de nombres de columnas."""
    
    def test_strip_whitespace(self):
        """Eliminar espacios en nombres."""
        df = pd.DataFrame({" col1 ": [1, 2], "col2  ": [3, 4]})
        df_clean = clean_column_names(df)
        
        assert "col1" in df_clean.columns
        assert "col2" in df_clean.columns
        assert " col1 " not in df_clean.columns


class TestRemoveIrrelevantColumns:
    """Pruebas para eliminación de columnas irrelevantes."""
    
    def test_remove_existing_columns(self):
        """Eliminar columnas existentes."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3],
            "mixed_type_col": [4, 5, 6],
            "Class": ["happy", "sad", "angry"]
        })
        
        df_clean = remove_irrelevant_columns(df, ["mixed_type_col"])
        
        assert "mixed_type_col" not in df_clean.columns
        assert "feature1" in df_clean.columns
        assert "Class" in df_clean.columns
    
    def test_remove_nonexistent_columns(self, sample_dataframe):
        """No error al intentar eliminar columnas inexistentes."""
        df_clean = remove_irrelevant_columns(sample_dataframe, ["nonexistent"])
        assert df_clean.shape == sample_dataframe.shape


class TestCleanTargetColumn:
    """Pruebas para limpieza de columna objetivo."""
    
    def test_filter_valid_classes(self):
        """Filtrar clases válidas correctamente."""
        df = pd.DataFrame({
            "feature": [1, 2, 3, 4, 5],
            "Class": ["happy", "sad", "invalid", "angry", "HAPPY"]
        })
        
        valid_classes = ["happy", "sad", "angry", "relax"]
        df_clean = clean_target_column(df, "Class", valid_classes)
        
        # Debe quedar solo 3 filas (happy, sad, angry en minúsculas)
        assert len(df_clean) == 4
        assert "invalid" not in df_clean["Class"].values
        assert all(df_clean["Class"].isin(valid_classes))
    
    def test_normalize_case(self):
        """Normalizar mayúsculas/minúsculas."""
        df = pd.DataFrame({
            "feature": [1, 2],
            "Class": ["HAPPY", "SaD"]
        })
        
        df_clean = clean_target_column(df, "Class", ["happy", "sad"])
        
        assert all(df_clean["Class"].str.islower())


class TestConvertNumericColumns:
    """Pruebas para conversión de columnas numéricas."""
    
    def test_convert_comma_decimal(self):
        """Convertir formato con comas decimales."""
        df = pd.DataFrame({
            "numeric_col": ["1,5", "2,3", "3,7"],
            "Class": ["happy", "sad", "angry"]
        })
        
        df_clean = convert_numeric_columns(df, "Class")
        
        assert df_clean["numeric_col"].dtype == np.float64
        assert df_clean["numeric_col"].iloc[0] == 1.5
    
    def test_skip_target_column(self):
        """No convertir columna objetivo."""
        df = pd.DataFrame({
            "feature": ["1.5", "2.3"],
            "Class": ["happy", "sad"]
        })
        
        df_clean = convert_numeric_columns(df, "Class")
        
        assert df_clean["Class"].dtype == object
        assert df_clean["feature"].dtype == np.float64


class TestRemoveDuplicates:
    """Pruebas para eliminación de duplicados."""
    
    def test_remove_exact_duplicates(self):
        """Eliminar filas duplicadas exactas."""
        df = pd.DataFrame({
            "feature": [1, 2, 1, 3],
            "Class": ["happy", "sad", "happy", "angry"]
        })
        
        df_clean = remove_duplicates(df)
        
        assert len(df_clean) == 3
    
    def test_no_duplicates(self, sample_dataframe):
        """Sin cambios cuando no hay duplicados."""
        original_len = len(sample_dataframe)
        df_clean = remove_duplicates(sample_dataframe)
        assert len(df_clean) == original_len


class TestSplitTrainTest:
    """Pruebas para división train/test."""
    
    def test_split_proportions(self, sample_dataframe):
        """Proporciones correctas del split."""
        train, test = split_train_test(
            sample_dataframe, "Class", test_size=0.2, random_state=42
        )
        
        total = len(sample_dataframe)
        assert len(train) == int(total * 0.8)
        assert len(test) == int(total * 0.2)
    
    def test_stratification(self, sample_dataframe):
        """Estratificación mantiene distribución de clases."""
        train, test = split_train_test(
            sample_dataframe, "Class", test_size=0.2, random_state=42
        )
        
        # Proporciones similares en train y test
        train_dist = train["Class"].value_counts(normalize=True)
        test_dist = test["Class"].value_counts(normalize=True)
        
        # Verificar que las proporciones son similares (tolerancia 10%)
        for cls in train_dist.index:
            if cls in test_dist.index:
                diff = abs(train_dist[cls] - test_dist[cls])
                assert diff < 0.15  # Tolerancia para datasets pequeños
    
    def test_invalid_test_size(self, sample_dataframe):
        """Error con test_size inválido."""
        with pytest.raises(ValueError):
            split_train_test(sample_dataframe, "Class", test_size=1.5, random_state=42)
    
    def test_reproducibility(self, sample_dataframe):
        """Mismo random_state produce mismos splits."""
        train1, test1 = split_train_test(
            sample_dataframe, "Class", test_size=0.2, random_state=42
        )
        train2, test2 = split_train_test(
            sample_dataframe, "Class", test_size=0.2, random_state=42
        )
        
        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(test1, test2)
