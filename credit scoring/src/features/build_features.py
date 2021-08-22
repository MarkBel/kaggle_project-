import numpy as np
import pandas as pd
import warnings
from datetime import datetime

warnings.filterwarnings(action="ignore")
from fe_utils import *


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    get preprocessed and merged with metadata train set or test set
    return dataframe with new features
    """

    df['LastPaymentMonth'] = df['LastPaymentDate'].dt.month
    df['FirstPaymentMonth'] = df['FirstPaymentDate'].dt.month
    df['pay_period'] = df['TransactionDates'].apply(lambda x: extract_payd_period(x))
    df['sum_delay_days'] = df['pay_period'].apply(lambda x: extract_delay_day(x))
    df['amount_delay_days'] = df['pay_period'].apply(lambda x: extract_amount_delays(x))
    df['amount_early_pays'] = df['pay_period'].apply(lambda x: extract_amount_early(x))
    df['amount_intime_pays'] = df['pay_period'].apply(lambda x: extract_amount_intime(x))
    df['mean_payment'] = df['PaymentsHistory'].apply(lambda x: np.mean(x))
    df['std_payment'] = df['PaymentsHistory'].apply(lambda x: np.std(x))
    df['max_payment'] = df['PaymentsHistory'].apply(lambda x: np.max(x))
    df['median_payment'] = df['PaymentsHistory'].apply(lambda x: np.median(x))
    df['min_payment'] = df['PaymentsHistory'].apply(lambda x: np.min(x))
    df['sum_payment'] = df['PaymentsHistory'].apply(lambda x: sum(x))
    df['RegisteredInLeapYear'] = df['RegistrationDate'].dt.is_leap_year.astype('float')
    df['RegisteredAtMonthStart'] = df['RegistrationDate'].dt.is_month_start.astype('float')
    df['RegisteredAtMonthEnd'] = df['RegistrationDate'].dt.is_month_end.astype('float')
    df['diff_expect_last'] = (df['ExpectedTermDate'] - df['LastPaymentDate']).dt.days
    df['real_duratation'] = (df['LastPaymentDate'] - df['FirstPaymentDate']).dt.days
    df['Reg_Duratation'] = (datetime.now() - df['RegistrationDate']).dt.days
    df['was_upsell'] = df['UpsellDate'].apply(lambda x: 1 if not pd.isnull(x) else 0)
    df['diff_real_expect'] = (df['Term'] - df['real_duratation'])
    interpl_targ(df)
    pad_history(df)

    df.drop(columns=[], inplace=True)
    return df


















