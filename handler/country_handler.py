import pandas as pd


def clean_country(df, num):
    """
    清洗国家数据 按国家分组算出拍过多少电影和平均分
    :param df: 处理前的dataframe
    :param num: 国家拍过的电影部数小于num的过滤
    :return: 处理过的dataframe
    """
    df = df.assign(country=df['country'].str.split('|')).explode('country')
    data = df.groupby(df['country']).mean().round(2).reset_index()
    df['cnt'] = 1
    df = df.groupby(['country'])['cnt'].size().reset_index(name='value')
    df = pd.merge(df, data, on='country')
    df.rename(columns={'country': 'name'}, inplace=True)
    df.drop(df[df['value'] < num].index, inplace=True)
    print(data)
    print(df)

    return df
