from common.api import api
import pandas as pd
from flask import Blueprint
from common.api import csv_url
from handler.rating_handler import clean_rating
from handler.budget_revenue_handler import clean_revenue
from handler.company_handler import clean_company
from handler.country_handler import clean_country
from handler.director_handler import clean_director

pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.2f" % x)

rating_bp = Blueprint('rating', __name__, url_prefix='/api/rating')


@rating_bp.route("/data", methods=["GET"])
def api_response():
    """
    :return: api数据
    """
    data_dic = {
        "ratingRevenueData": rating_revenue(),
        "ratingCompanyDateData": rating_company(),
        "ratingCountryNumData": rating_country_num(),
        "ratingDirectorNumData": rating_director_num(),
    }
    return api(data_dic)


def rating_revenue():
    df = pd.read_csv(csv_url, usecols=[10, 14])
    df.dropna(inplace=True)
    df = clean_rating(df)
    df = clean_revenue(df)
    data = df.to_dict("records")
    print(data)
    # print(df)

    return data


def rating_company():
    df = pd.read_csv(csv_url, usecols=[12, 14])
    df.dropna(inplace=True)
    df = clean_rating(df)
    data = clean_company(df)
    # data = df.sort_values(by='rating', ascending=False).to_dict("records")

    # print(data)
    # print(len(data))
    # print(df)

    return data


def rating_country_num():
    df = pd.read_csv(csv_url, usecols=[13, 14])
    df.dropna(inplace=True)
    df = clean_rating(df)
    df = clean_country(df, 50)

    data = {'name': 'root', 'children': df.to_dict("records")}
    # print(df)
    print(len(df))

    return data


def rating_director_num():
    df = pd.read_csv(csv_url, usecols=[6, 14])
    df.dropna(inplace=True)
    df = clean_rating(df)
    df = clean_director(df, 20)

    print(df)
    print(len(df))
    data = {'name': 'root', 'children': df.to_dict("records")}

    return data
