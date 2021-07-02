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
    df['RegisteredInLeapYear'] = df.RegistrationDate.dt.is_leap_year.astype('float')
    df['RegisteredAtMonthStart'] = df.RegistrationDate.dt.is_month_start.astype('float')
    df['RegisteredAtMonthEnd'] = df.RegistrationDate.dt.is_month_end.astype('float')
    df['LastPaymentMonth'] = df.LastPaymentDate.dt.month
    df['FirstPaymentMonth'] = df.FirstPaymentDate.dt.month
    df['pay_period'] = df['TransactionDates'].apply(lambda x:extract_payd_period(x))
    df['sum_delay_days'] = df['pay_period'].apply(lambda x:extract_delay_day(x))
    df['amount_delay_days'] = df['pay_period'].apply(lambda x:extract_amount_delays(x))
    df['amount_early_pays'] = df['pay_period'].apply(lambda x:extract_amount_early(x))
    df['amount_intime_pays'] = df['pay_period'].apply(lambda x:extract_amount_intime(x))
    df['mean_payment'] = df['PaymentsHistory'].apply(lambda x:np.mean(x))
    df['std_payment'] = df['PaymentsHistory'].apply(lambda x:np.std(x))
    df['max_payment'] = df['PaymentsHistory'].apply(lambda x:np.max(x))
    df['median_payment'] = df['PaymentsHistory'].apply(lambda x:np.median(x))
    df['min_payment'] = df['PaymentsHistory'].apply(lambda x:np.min(x))
    interpl_targ(df)
    pad_history(df)
    return df

















    
