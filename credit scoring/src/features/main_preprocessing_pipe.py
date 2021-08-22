import numpy as np
import pandas as pd
import pandasql as ps
import logging
from feature_extractor import create_features
from category_encoders.target_encoder import TargetEncoder
from category_encoders.m_estimate import MEstimateEncoder
from category_encoders.leave_one_out import LeaveOneOutEncoder
from category_encoders.cat_boost import CatBoostEncoder
from category_encoders.james_stein import JamesSteinEncoder
from category_encoders.one_hot import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from boostaroota import BoostARoota
# from visualize import process_data_per_feature_type, generate_drift_report
from feature_generator import FeatureGenerator



logging.basicConfig(filename='processing_log.txt', level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

ENCODERS_LIST: list = [OneHotEncoder(), TargetEncoder(), MEstimateEncoder(), JamesSteinEncoder(), LeaveOneOutEncoder(),
                       CatBoostEncoder()]

np.random.seed(8)


def extract_df_by_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def generate_col_on_multiple_conditions(df: pd.DataFrame, choices: np.ndarray) -> pd.DataFrame:
    """
    @param df: Dataframe to be extended
    @param choices: New categories to be assigned by custom rules. Ex. ['Senior','Senorita','Adult_male','Adult_Female']
    @return:
    """
    conditions = [
        (df['MainApplicantGender'] == 'Male') & (df['Age'] > 65),
        (df['MainApplicantGender'] == 'Female') & (df['Age'] > 65),
        (df['MainApplicantGender'] == 'Male') & (df['Age'] <= 65),
        (df['MainApplicantGender'] == 'Female') & (df['Age'] <= 65)
    ]
    df['status'] = np.select(conditions, choices, default=np.nan)
    logging.debug(f'Column status with values {choices} was generated.')
    return df


def generate_binary_feature_based_on_condition(df: pd.DataFrame, col_to_be_generated: str, col_name='Age',
                                               first_category='Senior', second_category='Adult') -> pd.DataFrame:
    """
    @param df: Dataframe to be extended
    @param col_to_be_generated: The new assigned name of the binary_col.
    @param col_name: Col to be processed.
    @param first_category: First new cat name.
    @param second_category: Second new cat name.
    @return:
    """
    df[col_to_be_generated] = np.where(df[col_name] > 50, first_category, second_category)
    logging.debug(f'Column {col_to_be_generated} with values first - {first_category} , second - {second_category}'
                  f' was generated.')
    return df


def dump_to_csv(df: pd.DataFrame, sep: str) -> None:
    df.to_csv(path_or_buf='processed_frame.csv', sep=sep)


def pipe_process_monitoring(df, fn=lambda x: x.shape, step=None, msg=None):
    """ Custom Help function to print things in method chaining.
        Returns back the df to further use in chaining.
    """
    if msg:
        print(msg)
    print(fn(df))
    logger.debug(f"Pipe monitoring is on the step {step}, df size is {fn(df)}")
    return df


def category_counter(df: pd.DataFrame, col_to_check: str) -> None:
    print(df[col_to_check].value_counts())


def create_custom_bins_digitize(df: pd.DataFrame) -> pd.DataFrame:
    bins = np.array([0, 20, 40, 60, 80, 100])
    df['Age'] = np.digitize(df.score_a, bins)
    # df['age_bin'] = pd.cut(df.age, 5)
    return df


def preprocess_frame(df: pd.DataFrame) -> pd.DataFrame:
    cleaned_df = (
        df.pipe(drop_cols_by_condition)
            .pipe(detect_unique_cols)
            .pipe(detect_constant_cols)
    )
    return cleaned_df


def drop_cols_by_condition(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [clmn for clmn in df.columns
                       if len(clmn) > 0]
    logging.debug(f'Unwanted columns_to_drop {columns_to_drop}')
    df.dropna()
    df = df.drop(columns_to_drop, axis=1).copy()
    return df


def detect_constant_cols(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [clmn for clmn in df.columns if len(df[clmn].value_counts()) < 2]
    logging.debug(f'Constant columns_to_drop {columns_to_drop}')
    df = df.drop(columns_to_drop, axis=1).copy()
    return df


def filter_out_columns_by_types(df: pd.DataFrame, filter_type: str) -> pd.DataFrame:
    global filtered_df
    for col, col_type in zip(df.columns, df.dtypes):
        if col_type == filter_type:
            logging.debug(f'Column that matching the time to be kept {col}')
        else:
            filtered_df = df.drop(col, axis=1).copy()
    return filtered_df


def detect_unique_cols(df: pd.DataFrame) -> pd.DataFrame:
    columns_unique = [clmn for clmn in df.columns if
                      len(df[clmn].value_counts()) == len(df)]
    logging.debug(f'Unique columns_to_drop {columns_unique}')
    df = df.drop(columns_unique, axis=1).copy()
    return df


def num_cols_processing(df: pd.DataFrame) -> list:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def cat_cols_processing(df: pd.DataFrame) -> list:
    return df.select_dtypes(include=[object]).columns.tolist()


def date_time_cols_processing(df: pd.DataFrame):
    for clmn in df.columns:
        possible_datetime = coerce_to_datetime_with_bounds(df[clmn])
    return possible_datetime


def coerce_to_datetime_with_bounds(potential_datetime_column, min_bound='1919-01-01', max_bound='2121-01-01'):
    result = pd.to_datetime(
        potential_datetime_column,
        format='%Y-%m-%d %H:%M:%S',
        errors='coerce')
    result[result < min_bound] = pd.NaT
    result[result > max_bound] = pd.NaT
    return result


def process_num_cols(df: pd.DataFrame) -> pd.DataFrame:
    num_cols_list = num_cols_processing(df)

    # TODO generate feature based on the target coverage
    for num_col in num_cols_list:
        num_clmn = df[num_col]


def process_non_num_cols(df: pd.DataFrame) -> list:
    non_num_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return non_num_columns


def detect_cat_cols(df: pd.DataFrame, upper_boundary: int) -> list:
    return [clmn for clmn in process_non_num_cols(df) if
            2 < len(df[clmn].value_counts()) < upper_boundary]


def detect_binary_cols(df: pd.DataFrame, ) -> list:
    return [clmn for clmn in process_non_num_cols(df) if
            len(df[clmn].value_counts()) == 2]


def possible_dates_to_datetime(df: pd.DataFrame, ) -> pd.DataFrame:
    potentially_datetime_reg_date = coerce_to_datetime_with_bounds(df['RegistrationDate'])
    potentially_datetime_last_date = coerce_to_datetime_with_bounds(df['LastPaymentDate'])
    potentially_datetime_first_date = coerce_to_datetime_with_bounds(df['FirstPaymentDate'])

    df['RegistrationDate'] = potentially_datetime_reg_date
    df['LastPaymentDate'] = potentially_datetime_last_date
    df['FirstPaymentDate'] = potentially_datetime_first_date
    return df


def merge_meta(df_transformed: pd.DataFrame, df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
    return pd.merge(left=df_transformed, right=pd.concat(objs=[df_train, df_test]).fillna(value=0.), on="ID")


def preprocess_payment_history(merged: pd.DataFrame) -> pd.DataFrame:
    merged['TransactionDates'] = merged['TransactionDates'].apply(lambda x: eval(x))
    merged['PaymentsHistory'] = merged['PaymentsHistory'].apply(lambda x: eval(x))
    merged['TransactionDates'] = merged['TransactionDates'].apply(lambda x: [
        pd.datetime.strptime(i, "%m-%Y").date() for i in x])
    return merged


def generate_col_with_sql_with_merge(df: pd.DataFrame, filter:str = None, created_name:str = None) -> pd.DataFrame:
    res = ps.sqldf(
        f"select age,count(*) as {created_name} from df where MainApplicantGender = \'{filter}\' group by age having count(*) >2")
    return pd.merge(df, res, on="Age")


if __name__ == '__main__':
    df_train = extract_df_by_path("../../data/01_raw/Train.csv")
    df_meta = extract_df_by_path("../../data/01_raw/metadata.csv")
    df_test = extract_df_by_path("../../data/01_raw/Test.csv")

    generate_col_with_sql_with_merge(df_meta, 'Male', 'Count_per_cat_male')
    generate_col_with_sql_with_merge(df_meta, 'Female', 'Count_per_cat_female')

    df_meta_train_preprocessed = (
        df_meta
            .pipe(possible_dates_to_datetime)
            .pipe(pipe_process_monitoring, step=1)
            .pipe(merge_meta, df_train, df_test)
            .pipe(pipe_process_monitoring, step=2)
            .pipe(preprocess_payment_history)
            .pipe(pipe_process_monitoring, step=3)
            .pipe(create_features)
            .pipe(pipe_process_monitoring, step=4)
            .pipe(generate_col_on_multiple_conditions, choices=['Senior', 'Senorita', 'Adult_male', 'Adult_Female'])
            .pipe(pipe_process_monitoring, step=5)
            .pipe(generate_binary_feature_based_on_condition, col_to_be_generated = 'Age_category')
            .pipe(pipe_process_monitoring, step=6)
            # .pipe(generate_col_with_sql_with_merge,filter='Male', created_name='Count_per_cat_male')
            # .pipe(generate_col_with_sql_with_merge, filter='Female', created_name='Count_per_cat_female')
            # .pipe(pipe_process_monitoring, step=7)
    )

    feature_generator = FeatureGenerator()
    feature_generator.build_knn(df_meta)

    ###### Evidently Data Drift report generation
    generate_drift_report(process_data_per_feature_type(df_meta_train_preprocessed),20000)
    ######

    # TODO Pipe Applying each of the transformers and evaluate the Loss
    for enc in ENCODERS_LIST:
        transformers_dict = {
            'encoder': enc,
            'imputer': StandardScaler(),
            'feature_eliminator': BoostARoota(metric='logloss')
        }
        X_train, y_train = df_meta_train_preprocessed.drop(columns=['ID', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6']), df_meta_train_preprocessed[
            ['m1', 'm2', 'm3', 'm4', 'm5', 'm6']]
        # transformers_dict.get('feature_eliminator').fit(X_train, y_train)
        # list_of_features_by_importance = transformers_dict.get('feature_eliminator').keep_vars_
        # transformers_dict.get('feature_eliminator').transform(X_train)
