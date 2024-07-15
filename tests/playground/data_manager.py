import pickle
from collections import namedtuple

import pandas as pd
from sklearn.model_selection import train_test_split

DIAGNOSIS_CODE = "icd_code"
ADMISSION_ID = "hadm_id"
PATIENT_ID = "subject_id"
ANALYSIS_ID = "itemid"
SPECIMEN_ID = "specimen_id"
ADMISSION_TIME = "admittime"
AGE = "anchor_age"
DIAGNOSIS_TIME = "diagnosis_time"
ANALYSIS_TIME = "storetime"
ANALYSIS_NAME = "label"
ANALYSIS_VALUE = "valuenum"
DATA_PATH = "C:/Biologie/mimic_data/matrices/"


def table_path(table_name):
    return f"C:/mimic_data/mimic_iv/mimic-iv-2.2/hosp/{table_name}.csv.gz"


def split_data(x, y, meta, test_size=0.2, random_state=None):
    """
    Split data between train and test set based on the patients' IDs.
    :return: x_train, x_test, y_train, y_test, meta_train, meta_test
    """
    df_id = pd.DataFrame([t[PATIENT_ID] for t in meta])

    train, test = train_test_split(df_id.drop_duplicates(), test_size=test_size, random_state=random_state)
    indices_train = pd.merge(df_id.reset_index(), train, on=0)["index"]
    indices_test = pd.merge(df_id.reset_index(), test, on=0)["index"]

    x_train = [x[i] for i in indices_train]
    x_test = [x[i] for i in indices_test]

    y_train = [y[i] for i in indices_train]
    y_test = [y[i] for i in indices_test]

    meta_train = [meta[i] for i in indices_train]
    meta_test = [meta[i] for i in indices_test]

    return x_train, x_test, y_train, y_test, meta_train, meta_test


def get_demographics(analysis_metadata, patients_table):
    """Given the metadata of an analysis (subject_id and storetime),
    computes the age of the corresponding patient at the time of the analysis.
    Does not account for exact birthdate, but a +/-1 year difference is accepted"""
    # FIXME we assume the id is effectively in the patients table
    row = patients_table[patients_table[PATIENT_ID] == analysis_metadata[PATIENT_ID]].iloc[0][
        ["gender", "anchor_age", "anchor_year"]]
    current_age = (
            row["anchor_age"] +  # age used as reference
            pd.to_datetime(analysis_metadata[ANALYSIS_TIME]).year -  # year when the analysis was performed
            pd.to_datetime(row["anchor_year"], format="%Y").year  # year used as reference
    )
    return {"gender": row["gender"], "age": current_age}


def load_diagnoses_lookup():
    return pd.read_csv(table_path("d_icd_diagnoses"))


def load_analyses_lookup():
    return pd.read_csv(table_path("d_labitems"))


def load_data(dataset_name, random_state):
    """
    :return: Biological data retrieved from the given dataset, split between train and test set.
    """
    # Load the relevant tables
    patients = pd.read_csv(table_path("patients"))
    STORAGE = dataset_name + "/"

    with open(DATA_PATH + STORAGE + 'positive_patients_matrices.pkl', 'rb') as handle:
        x_pos, y_pos, meta_pos = pickle.load(handle)
    with open(DATA_PATH + STORAGE + 'negative_patients_matrices.pkl', 'rb') as handle:
        x_neg, y_neg, meta_neg = pickle.load(handle)

    # Split between training and test set
    x_train_pos, x_test_pos, y_train_pos, y_test_pos, meta_train_pos, meta_test_pos \
        = split_data(x_pos, y_pos, meta_pos, random_state=random_state)
    x_train_neg, x_test_neg, y_train_neg, y_test_neg, meta_train_neg, meta_test_neg \
        = split_data(x_neg, y_neg, meta_neg, random_state=random_state)

    # Rejoin positive and negative samples
    bio_data_train, labels_train, meta_train \
        = x_train_pos + x_train_neg, y_train_pos + y_train_neg, meta_train_pos + meta_train_neg
    bio_data_test, labels_test, meta_test \
        = x_test_pos + x_test_neg, y_test_pos + y_test_neg, meta_test_pos + meta_test_neg

    return bio_data_test, bio_data_train, labels_test, labels_train, meta_test, meta_train, patients


def load_stage_5_ckd(random_state=None):
    bio_data_test, bio_data_train, labels_test, labels_train, meta_test, meta_train, patients \
        = load_data("stage_5_ckd", random_state)

    # Create data sources
    data_source_1_train = [pd.DataFrame(t["analysis_50912"]) for t in bio_data_train]  # creatinine measurements
    data_source_2_train = pd.DataFrame([get_demographics(t, patients) for t in meta_train])

    data_source_1_test = [pd.DataFrame(t["analysis_50912"]) for t in bio_data_test]  # creatinine measurements
    data_source_2_test = pd.DataFrame([get_demographics(t, patients) for t in meta_test])

    LearningData = namedtuple(
        typename="LearningData",
        field_names=["data_sources_train", "labels_train", "data_sources_test", "labels_test"]
    )

    return LearningData(
        data_sources_train=[data_source_1_train, data_source_2_train],
        labels_train=labels_train,
        data_sources_test=[data_source_1_test, data_source_2_test],
        labels_test=labels_test,
    )
