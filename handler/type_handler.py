import pandas as pd


def clean_type_numeric(df, col):
    # 分割每个type出新的一行
    df = clean_type(df)
    df[col] = pd.to_numeric(df[col])
    data = df.groupby(df['type']).mean().round(2)
    data = data.reset_index()

    return data


def clean_type(df):
    df = df.assign(type=df['type'].str.split('|')).explode('type')

    return df
