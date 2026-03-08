import pandas as pd
from src.housing_utils import prepare_features


def test_prepare_features_creates_expected_columns():
    df = pd.DataFrame({
        "price": [100000, 200000],
        "area": [50, 100],
        "floor": [2, 5],
        "number_of_floors": [10, 10]
    })

    result = prepare_features(df)

    assert "log_price" in result.columns
    assert "log_area" in result.columns
    assert "price_per_sqm" in result.columns
    assert "floor_ratio" in result.columns