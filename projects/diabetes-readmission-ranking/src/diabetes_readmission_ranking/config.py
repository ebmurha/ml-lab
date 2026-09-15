from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_PATH = DATA_DIR / "diabetic_data.csv"
ID_MAPPING_PATH = DATA_DIR / "IDS_mapping.csv"

RANDOM_SEED = 42
TARGET_COLUMN = "readmitted"
PATIENT_ID_COLUMN = "patient_nbr"
ENCOUNTER_ID_COLUMN = "encounter_id"
FOLLOW_UP_CAPACITY = 0.10

TRAIN_FRACTION = 0.70
CALIBRATION_FRACTION = 0.15
TEST_FRACTION = 0.15
CV_FOLDS = 3

# Patients who died or entered hospice are not candidates for readmission outreach.
INELIGIBLE_DISCHARGE_DISPOSITIONS = frozenset({11, 13, 14, 19, 20, 21})

FORBIDDEN_FEATURES = frozenset(
    {ENCOUNTER_ID_COLUMN, PATIENT_ID_COLUMN, TARGET_COLUMN, "target"}
)

CATEGORICAL_CODE_COLUMNS = frozenset(
    {"admission_type_id", "discharge_disposition_id", "admission_source_id"}
)

MODEL_PATH = ARTIFACTS_DIR / "readmission_ranker.joblib"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"
