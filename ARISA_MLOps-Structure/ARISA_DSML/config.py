"""Config file for module."""
from pathlib import Path
from plotly import graph_objs as go
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATASET = "cholesterol-supplementation-classification"  # original dataset
DATASET_TEST = "cholesterol-supplementation-classification"
# test set augmented with target labels

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODEL_NAME = "cholesterol-pred-bclass"

categorical = [
    "Gender",
    "Physical_Activity",
    "Dietary_Habits",
    "Family_History",
]

target = "Need_Supplement"
