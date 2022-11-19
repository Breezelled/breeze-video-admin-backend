import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def clean_budget_revenue_exception(df):
    # 清洗异常值
    # print(len(df[(df['revenue'] - df['budget']) > (df['revenue'] - df['budget']).mean()*20]))
    # print(len(df[(df['revenue'] - df['budget']) < (df['budget'] - df['revenue']).mean()*20]))
    df.drop(df[(df['revenue'] - df['budget']) > (df['revenue'] - df['budget']).mean() * 20].index, inplace=True)
    df.drop(df[(df['revenue'] - df['budget']) < (df['budget'] - df['revenue']).mean() * 20].index, inplace=True)
    # print((df['revenue'] - df['budget']).mean())
    return df


def clean_budget(df):
    # df['budget'] = df['budget'].str.replace(r'\(estimated\)', '')
    df['budget'] = df['budget'].str.replace('£', '$')
    df['budget'] = df['budget'].str.replace('€', '$')
    df.drop(df[df['budget'].str[0] != '$'].index, inplace=True)
    # df.drop(df[~df['budget'].str.contains(r'\$')].index, inplace=True)
    # 转换为数字
    df['budget'] = df['budget'].str.replace(',', '')
    df['budget'] = df['budget'].str.extract(r'(\d+)', expand=False)
    df['budget'] = pd.to_numeric(df['budget'])
    return df


def clean_revenue(df):
    # 转换为数字
    df['revenue'] = df['revenue'].str.replace(',', '')
    df['revenue'] = df['revenue'].str.extract(r'(\d+)', expand=False)
    df['revenue'] = pd.to_numeric(df['revenue'])
    return df


def budget_revenue(df):
    sns.lmplot(x="budget", y="revenue", data=df)
    plt.title("电影预算与票房关系图")
    plt.xlabel("电影预算")
    plt.ylabel("电影票房")


def vote_revenue(df):
    sns.lmplot(x="vote_average", y="revenue", data=df)
    plt.title("电影评分与票房关系图")
    plt.xlabel("电影评分")
    plt.ylabel("电影票房")


def revenue(df):
    res_df = pd.DataFrame([df["revenue"], df["release_date"]]).T
    res_df["release_date"] = pd.DatetimeIndex(df["release_date"]).year
    return res_df


def annual_revenue(df):
    res_df = revenue(df)
    sns.lineplot(x="release_date", y="revenue", data=res_df, hue="")
    plt.title("每年总票房统计")
    plt.xlabel("年份")
    plt.ylabel("总票房")


def search_revenue(df, start, end):
    res_df = revenue(df)
    if start < res_df["release_date"].min():
        start = res_df["release_date"].min()
    if end > res_df["release_date"].max():
        end = res_df["release_date"].max()
    res_df = res_df[res_df["release_date"] >= start]
    res_df = res_df[res_df["release_date"] <= end]
    sns.lineplot(x="release_date", y="revenue", data=res_df)
    plt.title(str(start) + "年至" + str(end) + "年总票房统计图")
    plt.xlabel("年份")
    plt.ylabel("总票房")


def num(df):
    res_df = pd.DataFrame(df["release_date"])
    res_df["release_date"] = pd.DatetimeIndex(df["release_date"]).year
    res_df = pd.DataFrame(res_df["release_date"].value_counts())
    res_df["date"] = res_df.index
    res_df.columns = ["count", "date"]
    res_df.index = range(len(res_df))
    return res_df


def movie_num(df):
    res_df = num(df)
    sns.lineplot(x="date", y="count", data=res_df)
    plt.title("每年上映电影数量统计图")
    plt.xlabel("年份")
    plt.ylabel("上映电影数量")


def search_movie_num(df, start, end):
    res_df = num(df)
    if start < res_df["date"].min():
        start = res_df["date"].min()
    if end > res_df["date"].max():
        end = res_df["date"].max()
    res_df = res_df[res_df["date"] >= start]
    res_df = res_df[res_df["date"] <= end]
    sns.lineplot(x="date", y="count", data=res_df)
    plt.title(str(start) + "年至" + str(end) + "年上映电影数量统计图")
    plt.xlabel("年份")
    plt.ylabel("上映电影数量")


def main():
    # df = pd.read_csv("info.csv", usecols=[9, 10])
    print()
    # sns.set_style(rc={'font.sans-serif': "Arial Unicode MS"})
    # df = pd.read_csv("tmdb_5000_movies.csv", )
    # budget_revenue(df)  # 预算和票房
    # vote_revenue(df)  # 评分和票房
    # annual_revenue(df)  # 每年总票房
    # search_revenue(df, 1940, 2000)  # 设置区间的每年票房
    # movie_num(df)  # 每年电影数量
    # search_movie_num(df, 0, 20000)  # 设置区间的每年电影数量
    # g2plot_test()
    # plt.show()
