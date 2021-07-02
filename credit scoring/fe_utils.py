import numpy as np
import pandas as pd 
import warnings
from typing import List
warnings.filterwarnings(action="ignore")

def extract_payd_period(x: List) -> List:
    """
    extract period between 2 payments
    """
    lst = []
    for i in range(1, len(x)):
        delta = (x[i] - x[i - 1]).days
        lst.append(delta)
    return lst

def extract_delay_day(x: List) -> int:
    """
    extract how many days are overdue during the loan
    """
    count = 0
    for i in x:
        if i > 31:
            count += i - 30
    return count

def extract_amount_delays(x: List) -> int:
    """
    extract how many delays in payment 
    """
    count = 0
    for i in x:
        if i > 31:
            count += 1
    return count

def extract_amount_early(x: List) -> int:
    """
    extract how much did pay more often than once a month 
    """
    count = 0
    for i in x:
        if i < 28:
            count += 1
    return count

def extract_amount_intime(x: List) -> int:
    """
    extract how much did you pay on time 
    """
    count = 0
    for i in x:
        if i > 27 and i < 32:
            count += 1
    return count

def interpl_targ(df: pd.DataFrame) -> None:
    """
    interpolation of target by simple rules
    """
    window = 2
    df["ExpectedTermDate"] = pd.to_datetime(df["ExpectedTermDate"]) + pd.to_timedelta(pd.np.ceil(df.Term*window), unit="D")
    df["LastPaymentDate"] = pd.to_datetime(df["LastPaymentDate"]).dt.tz_localize(None)
    culc_fild = []
    for r in df.iterrows():
        r = r[1]
        culc = ((100 * sum((r.PaymentsHistory) + r[-6:].values.tolist()) // r.TotalContractValue)>=60.) and (r.LastPaymentDate < r.ExpectedTermDate)
        culc_fild.append(float(culc))

    df["culc_fild"] = culc_fild
    

def pad_history(df:pd.DataFrame, max_len:int=41) -> None:
    """

    """
    padded_payments = []
    
    for r in df.copy().iterrows():
        r = r[1]
        
        if len(r.PaymentsHistory) > max_len:
            padded_payments.append(r.PaymentsHistory[:max_len])
            
        else:
            padding_len = abs(max_len - len((r.PaymentsHistory)))
            padded_payments.append(r.PaymentsHistory + padding_len*[0.])
            
    
    df["PaymentsHistory"] = padded_payments


def create_extra_features(df: pd.DataFrame) -> None:
    """
    count of nans
    """
    df['NANs_cnt'] = df.isnull().sum(axis = 1)
    
def create_col_with_min_freq(data, col, min_freq = 10):
    """
    cluster rare catetories in col
    """
    data[col + '_fixed'] = data[col].astype(str)
    data.loc[data[col + '_fixed'].value_counts()[data[col + '_fixed']].values < min_freq, col + '_fixed'] = "RARE_VALUE"
    data.replace({'nan': np.nan}, inplace = True)

def create_gr_feats(data,cat_cols,num_cols):
    """
    get statistics over cat_cols
    """
    for cat_col in cat_cols:
        create_col_with_min_freq(data, cat_col, 25)
        for num_col in num_col:
            for n, f in [('mean', np.mean), ('min', np.nanmin), ('max', np.nanmax)]:
                data['FIXED_' + n + '_' + num_col + '_by_' + cat_col] = data.groupby(cat_col + '_fixed')[num_col].transform(f)
    for col in cat_cols+num_cols:
        data[col + '_cnt'] = data[col].map(data[col].value_counts(dropna = False))
