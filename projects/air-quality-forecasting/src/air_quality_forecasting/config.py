"""Shared forecasting configuration."""

HORIZON_HOURS = 6
HOLDOUT_DAYS = 30
VALIDATION_WINDOWS = 4
RANDOM_STATE = 42

SIGNAL_COLUMNS = ("pm25", "temperature", "relativehumidity")
LAG_HOURS = (1, 3, 6, 12, 18, 24, 48, 72, 168)
ROLLING_HOURS = (3, 6, 12, 24, 72, 168)

MODEL_PARAMETERS = {
    "learning_rate": 0.05,
    "max_iter": 300,
    "max_leaf_nodes": 31,
    "l2_regularization": 1.0,
    "random_state": RANDOM_STATE,
}
