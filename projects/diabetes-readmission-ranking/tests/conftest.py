import pandas as pd
import pytest


@pytest.fixture
def encounters() -> pd.DataFrame:
    rows = []
    for patient in range(40):
        for visit in range(2):
            positive = patient % 4 == 0 and visit == 1
            rows.append(
                {
                    "encounter_id": patient * 10 + visit,
                    "patient_nbr": patient,
                    "race": "Group A" if patient % 2 else "Group B",
                    "gender": "Female" if patient % 2 else "Male",
                    "age": "[50-60)" if patient % 2 else "[60-70)",
                    "admission_type_id": 1,
                    "discharge_disposition_id": 1,
                    "admission_source_id": 7,
                    "time_in_hospital": visit + 1,
                    "num_medications": 5 + patient % 3,
                    "readmitted": "<30" if positive else "NO",
                }
            )
    return pd.DataFrame(rows)
