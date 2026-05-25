"""
Unit tests for the data_manager service.
Run with: pytest tests/
"""
import pytest


def test_convert_numpy_types_int():
    import numpy as np
    from app.utils.data_manager import convert_numpy_types

    result = convert_numpy_types(np.int64(42))
    assert result == 42
    assert isinstance(result, int)


def test_convert_numpy_types_float():
    import numpy as np
    from app.utils.data_manager import convert_numpy_types

    result = convert_numpy_types(np.float64(3.14))
    assert abs(result - 3.14) < 1e-9
    assert isinstance(result, float)


def test_convert_numpy_types_nested():
    import numpy as np
    from app.utils.data_manager import convert_numpy_types

    data = {"price": np.int64(500000), "scores": [np.float64(4.5), np.float64(3.0)]}
    result = convert_numpy_types(data)
    assert result == {"price": 500000, "scores": [4.5, 3.0]}
