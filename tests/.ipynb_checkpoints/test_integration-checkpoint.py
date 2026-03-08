import pandas as pd
from src.housing_utils import prepare_features, fit_baseline_model


def test_pipeline_runs_end_to_end():
    df = pd.DataFrame({
        "price": [100000, 150000, 200000, 250000, 300000],
        "area": [40, 50, 60, 70, 80],
        "minutes_to_metro": [5, 10, 12, 15, 20],
        "region": ["Moscow", "Moscow", "Moscow region", "Moscow", "Moscow region"],
        "apartment_type": ["Secondary", "New building", "Secondary", "New building", "Secondary"],
        "renovation": ["Cosmetic", "Designer", "Without renovation", "European-style renovation", "Cosmetic"],
        "floor": [2, 5, 3, 8, 4],
        "number_of_floors": [10, 12, 9, 16, 8]
    })

    df = prepare_features(df)
    model = fit_baseline_model(df)

    assert model is not None
    assert len(model.params) > 0