import pandas as pd
from datetime import datetime

def preprocess_data(df: pd.DataFrame) -> None:
    df['TransactionDates'] = df['TransactionDates'].apply(lambda x: eval(x))
    df['PaymentsHistory'] = df['PaymentsHistory'].apply(lambda x: eval(x))
    df['TransactionDates'] = df['TransactionDates'].apply(lambda x: [datetime.strptime(i, "%m-%Y").date() for i in x])
    df['RegistrationDate'] = pd.to_datetime(df['RegistrationDate'])
    df['LastPaymentDate'] = pd.to_datetime(df['LastPaymentDate'])
    df['FirstPaymentDate'] = pd.to_datetime(df['FirstPaymentDate'])
    df['ExpectedTermDate'] = pd.to_datetime(df['ExpectedTermDate'])

def filling_nulls(df: pd.DataFrame) -> None: 
    df['Region'].fillna(value=df.Region.mode()[0], inplace=True)
    df['Age'].fillna(value=round(df.Age.mean()), inplace=True)