from diabetes_readmission_ranking.data import eligible_encounters, split_by_patient


def test_ineligible_outcomes_are_removed(encounters):
    encounters.loc[0, "discharge_disposition_id"] = 11
    eligible = eligible_encounters(encounters)
    assert 0 not in set(eligible["encounter_id"])
    assert set(eligible["target"]) == {0, 1}


def test_patients_never_cross_partitions(encounters):
    partitions = split_by_patient(eligible_encounters(encounters))
    patients = [set(frame["patient_nbr"]) for frame in partitions.__dict__.values()]
    assert not patients[0] & patients[1]
    assert not patients[0] & patients[2]
    assert not patients[1] & patients[2]
