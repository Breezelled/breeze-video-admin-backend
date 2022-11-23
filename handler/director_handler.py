import pandas as pd


def clean_director(df, num):
    """
    清洗导演数据 按导演分组算出拍过多少电影和平均分
    :param df: 处理前的dataframe
    :param num: 导演拍过的电影部数小于num的过滤
    :return: 处理过的dataframe
    """
    data = df.groupby(df['director']).mean().round(2).reset_index()
    df['cnt'] = 1
    df = df.groupby(['director'])['cnt'].size().reset_index(name='value')
    df = pd.merge(df, data, on='director')
    df.rename(columns={'director': 'name'}, inplace=True)
    df.drop(df[df['value'] < num].index, inplace=True)

    return df
