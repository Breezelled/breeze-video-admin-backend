import pandas as pd


def clean_rating(df):
    df['rating'] = df['rating'].str.extract(r'(\d\.\d)', expand=False)
    df['rating'] = pd.to_numeric(df['rating'])

    return df
