from common.api import api
import pandas as pd
from flask import Blueprint
from handler.budget_revenue_handler import clean_revenue, clean_budget_revenue_exception, \
    clean_budget
from handler.release_date_handler import clean_date_to_year
from common.api import csv_url

pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.2f" % x)

budget_revenue_bp = Blueprint('budget_revenue', __name__, url_prefix='/api/budget_revenue')


@budget_revenue_bp.route("/data", methods=["GET"])
def api_response():
    """
    :return: api数据
    """
    data_dic = {
        "budgetRevenueData": budget_revenue(),
        "budgetData": budget(),
        "revenueData": revenue(),
    }
    return api(data_dic)


def budget_revenue():
    df = pd.read_csv(csv_url, usecols=[9, 10])
    # 删除空值
    df.dropna(inplace=True)
    # 清洗数据
    df = clean_budget(df)
    df = clean_revenue(df)
    df = clean_budget_revenue_exception(df)

    print(df)
    print(len(df))
    data = df.to_dict("records")
    # print(budget_revenue_data)
    return data


def budget():
    df = pd.read_csv(csv_url, usecols=[4, 9])
    df.dropna(inplace=True)
    df = clean_budget(df)
    df = clean_date_to_year(df)
    data = df.groupby(df['release_date']).mean().round(2).reset_index().to_dict("records")

    return data


def revenue():
    df = pd.read_csv(csv_url, usecols=[4, 10])
    df.dropna(inplace=True)
    df = clean_revenue(df)
    df = clean_date_to_year(df)
    data = df.groupby(df['release_date']).mean().round(2).reset_index().to_dict("records")

    return data
