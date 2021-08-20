import pandas as pd

from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab
#from evidently.tabs import CatTargetDriftTab


def process_data_per_feature_type(meta_frame: pd.DataFrame, col_type_requested: str = 'float', col_type_requested_second: str = None) -> pd.DataFrame:
    for col, col_type in zip(meta_frame.columns, meta_frame.dtypes):
        if (col_type == col_type_requested) or (col_type == col_type_requested_second):
            print(f'It is {col} of type {col_type_requested}. Keeping')
        else:
            meta_frame = meta_frame.drop(col, axis=1).copy()
    return  meta_frame


def generate_drift_report(df: pd.DataFrame, requested_lenght: int = 10000):
    credit_data_drift_report = Dashboard(tabs=[DataDriftTab])
    credit_data_drift_report.calculate(df[:requested_lenght], df[:requested_lenght], column_mapping=None)
    credit_data_drift_report.save("../../reports/my_report_numerical_features.html")

def generate_drift_report_with_mapping(df: pd.DataFrame, requested_mapping_list: list, requested_lenght: int = 10000):
    column_mapping = {}
    column_mapping['target'] = None
    column_mapping['numerical_features'] = None
    column_mapping['categorical_features'] = requested_mapping_list
    # column_mapping['categorical_features'] = ['rateTypeEntity', 'Town', 'Region']
    credit_data_drift_report = Dashboard(tabs=[DataDriftTab])
    credit_data_drift_report.calculate(df[:requested_lenght], df[:requested_lenght], column_mapping=column_mapping)
    credit_data_drift_report.save("../../reports/my_report_categorical.html")

