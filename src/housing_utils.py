import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


def load_data(path: str) -> pd.DataFrame:
    """Load the Moscow housing dataset from CSV."""
    df = pd.read_csv(path)
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create transformed variables used for analysis and modelling."""
    df = df.copy()
    df["log_price"] = np.log(df["price"])
    df["log_area"] = np.log(df["area"])
    df["price_per_sqm"] = df["price"] / df["area"]
    df["floor_ratio"] = df["floor"] / df["number_of_floors"]
    return df


def plot_area_vs_price(df: pd.DataFrame) -> None:
    """Plot log area against log price."""
    plt.figure(figsize=(8, 5))
    plt.scatter(df["log_area"], df["log_price"], alpha=0.2)
    plt.title("Log Apartment Area vs Log Price in Moscow")
    plt.xlabel("Log Area")
    plt.ylabel("Log Price")
    plt.show()


def plot_price_by_renovation(df: pd.DataFrame) -> None:
    """Plot price per square meter by renovation category."""
    plt.figure(figsize=(9, 5))
    df.boxplot(column="price_per_sqm", by="renovation", grid=False)
    plt.title("Price per Square Meter by Renovation")
    plt.suptitle("")
    plt.xlabel("Renovation")
    plt.ylabel("Price per square meter")
    plt.xticks(rotation=20)
    plt.show()


def fit_baseline_model(df: pd.DataFrame):
    """Fit the baseline OLS model for log housing prices."""
    model = smf.ols(
        formula="""
        log_price ~ log_area
                  + minutes_to_metro
                  + C(region)
                  + C(apartment_type)
                  + C(renovation)
                  + floor_ratio
        """,
        data=df
    ).fit()
    return model


# We exclude living_area, kitchen_area, and number_of_rooms from the baseline model
# because they overlap strongly with total area and may create multicollinearity.