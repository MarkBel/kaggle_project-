import pandas as pd
from datetime import datetime
from tqdm import tqdm

def preprocess_data(df: pd.DataFrame) -> None:
    df['TransactionDates'] = df['TransactionDates'].apply(lambda x: eval(x))
    df['PaymentsHistory'] = df['PaymentsHistory'].apply(lambda x: eval(x))
    df['TransactionDates'] = df['TransactionDates'].apply(lambda x: [datetime.strptime(i, "%m-%Y").date() for i in x])


def preprocess_metadata(df: pd.DataFrame) -> None:
    cols = [col for col in df.columns[1:]]
    cat_cols = [col for col in cols if df[col].dtype == 'O' and 'Date' not in col]
    date_cols = [col for col in cols if 'Date' in col]

    # or another cat encoding 
    for col in tqdm(cat_cols, desc='Processing categorical columns \t'):
        df[col] = df[col].astype('category')
    
    for col in tqdm(date_cols, desc='Processing datetime columns \t'):
        df[col] = pd.to_datetime(df[col]).dt.tz_localize(None)


def filling_nulls(df: pd.DataFrame) -> None: 
    df['Region'].fillna(value=df.Region.mode()[0], inplace=True)
    df['Age'].fillna(value=round(df.Age.mean()), inplace=True)