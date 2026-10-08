"""Configuration constants, column definitions, and business assumptions."""

from pathlib import Path

# Base Paths (Windows-safe pathlib)
ROOT_DIR: Path = Path(__file__).resolve().parent.parent.parent
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DATA_PATH: Path = DATA_DIR / "raw" / "Churn_Modelling.csv"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
TRAIN_DATA_PATH: Path = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH: Path = PROCESSED_DATA_DIR / "test.csv"
REPORTS_DIR: Path = ROOT_DIR / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
DATA_SUMMARY_PATH: Path = REPORTS_DIR / "data_summary.json"
METRICS_PATH: Path = REPORTS_DIR / "metrics.json"
MODELS_DIR: Path = ROOT_DIR / "models"
MODEL_PATH: Path = MODELS_DIR / "churn_model.joblib"
METADATA_PATH: Path = MODELS_DIR / "metadata.json"

# Random Seed & Target
SEED: int = 42
TARGET: str = "Exited"

# Column Groups
DROP_COLS: list[str] = ["RowNumber", "CustomerId", "Surname"]
NUMERIC: list[str] = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "EstimatedSalary",
]
BINARY: list[str] = ["HasCrCard", "IsActiveMember"]
CATEGORICAL: list[str] = ["Geography", "Gender"]
ALL_FEATURE_COLS: list[str] = NUMERIC + BINARY + CATEGORICAL

# Categorical Value Sets
VALID_GEOGRAPHIES: list[str] = ["France", "Germany", "Spain"]
VALID_GENDERS: list[str] = ["Female", "Male"]

# Business Assumptions (defaults)
CLV_RETAINED: float = 500.0  # value of keeping a customer
OFFER_COST: float = 50.0  # cost of a retention offer
OFFER_SUCCESS_RATE: float = 0.30  # offer acceptance/retention success rate
