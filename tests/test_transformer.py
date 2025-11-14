import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from eq18_turkish_music_mlops.utils.transformers import clean_finite_values

def test_clean_finite_values():
    """
    Prueba unitaria para la función 'clean_finite_values'.
    """

    dirty_data = np.array([
        [1.0, 5.0, np.nan],
        [4.0, np.inf, 6.0],
        [7.0, -np.inf, 9.0]
    ])
    
    clean_data = clean_finite_values(dirty_data)
    
    assert not np.isnan(clean_data).any(), "La función no eliminó los NaN"
    
    assert np.isfinite(clean_data).all(), "La función no eliminó los Infinitos (inf o -inf)"
    
    assert 1.0 in clean_data
    assert 9.0 in clean_data