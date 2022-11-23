import pandas as pd


def clean_country(df):
    df = df.assign(country=df['country'].str.split('|')).explode('country')
    data = df.groupby(df['country']).mean().round(2).reset_index()
    df['cnt'] = 1
    df = df.groupby(['country'])['cnt'].size().reset_index(name='value')
    df = pd.merge(df, data, on='country')
    print(data)
    print(df)

    return df
