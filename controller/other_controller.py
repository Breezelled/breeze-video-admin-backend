from common.api import api
import pandas as pd
from flask import Blueprint
from handler.company_handler import clean_company
from handler.type_handler import clean_type
from common.api import info_csv_url, freq_words_csv_url, review_csv_url

pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.2f" % x)

other_bp = Blueprint('other', __name__, url_prefix='/api/other')


@other_bp.route("/data", methods=["GET"])
def api_response():
    """
    :return: api数据
    """
    data_dic = {
        "runtimeData": runtime(),
        "reviewWordFrequencyData": review_word_frequency(),
        "companyMovieNumData": company_movie_num(),
        "companyTypeProportionData": company_type_proportion(),
        "starNumData": star_num(),
        "reviewerNumData": reviewer_num(),
    }
    return api(data_dic)


def runtime():
    df = pd.read_csv(info_csv_url, usecols=[2])
    df.dropna(inplace=True)
    df['runtime'] = (pd.to_timedelta(df.runtime).dt.total_seconds() / 60).astype(int)
    short = len(df[df['runtime'] < 90])
    medium = len(df[(df['runtime'] >= 90) & (df['runtime'] <= 120)])
    long = len(df[df['runtime'] > 120])
    # print(df)
    # print(len(df))
    # print(short)
    # print(medium)
    # print(long)
    # print(short+medium+long)
    data = [
        {
            "type": "短时长",
            "count": short,
        },
        {
            "type": "中等时长",
            "count": medium,
        },
        {
            "type": "长时长",
            "count": long,
        },
    ]

    return data


def review_word_frequency():
    df = pd.read_csv(freq_words_csv_url, usecols=[0, 1])
    data = df.to_dict("records")
    # print(data)

    return data


def company_movie_num():
    df = pd.read_csv(info_csv_url, usecols=[12])
    df.dropna(inplace=True)
    df = clean_company(df)
    df = df.groupby(df['company']).size().reset_index(name="count")
    df = df.sort_values(by='count', ascending=True)
    data = df.to_dict("records")
    # print(df)

    return data


def company_type_proportion():
    df = pd.read_csv(info_csv_url, usecols=[3, 12])
    df.dropna(inplace=True)
    df = clean_company(df)
    df = clean_type(df)
    top7_type = ['Drama', 'Comedy', 'Action', 'Romance', 'Crime', 'Adventure', 'Thriller']
    df = df.groupby(['company', 'type']).size().reset_index(name="count")
    top_data = df.loc[df['type'].isin(top7_type) & df['company'].isin(df['company'].unique())]
    other_data = df.loc[~df['type'].isin(top7_type)].groupby(['company'])['count'].sum().reset_index(name="count")
    other_data['type'] = 'Other'
    other_data = other_data.loc[:, ['company', 'type', 'count']]
    # print(other_data)
    data = top_data.append(other_data)
    data = data.to_dict("records")
    # print(data)

    return data


def star_num():
    df = pd.read_csv(info_csv_url, usecols=[8])
    df.dropna(inplace=True)
    df = df.assign(star=df['star'].str.split('|')).explode('star')
    df = df.groupby(['star']).size().reset_index(name="count")
    data = df.sort_values(by='count', ascending=False).head(20)
    data = data.sort_values(by='count')
    print(data)
    data = data.to_dict("records")

    return data


def reviewer_num():
    df = pd.read_csv(review_csv_url, usecols=[1])
    df.dropna(inplace=True)
    df = df.groupby(['author']).size().reset_index(name="count")
    data = df.sort_values(by='count', ascending=False).head(20)
    data = data.sort_values(by='count')
    print(data)
    data = data.to_dict("records")

    return data
