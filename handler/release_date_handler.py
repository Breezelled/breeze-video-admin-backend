import pandas as pd


def clean_date_to_year(df):
    df['release_date'] = df['release_date'].str.extract(r'(\d\d\d\d)', expand=False)

    return df