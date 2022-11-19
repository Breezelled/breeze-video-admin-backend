from common.api import api
import pandas as pd
from flask import Blueprint
from handler.budget_revenue_handler import clean_budget, clean_revenue, clean_budget_revenue_exception, clean_budget

pd.set_option("display.max_rows", None)

budget_revenue_bp = Blueprint('budget_revenue', __name__, url_prefix='/api/budget_revenue')

csv_url = "./IMDB/info/info.csv"


@budget_revenue_bp.route("/budget_revenue", methods=["GET"])
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
    # print(data)
    return api(data)


@budget_revenue_bp.route("/type_budget", methods=["GET"])
def type_budget():
    df = pd.read_csv(csv_url, usecols=[3, 9])
    df.dropna(inplace=True)
    df = clean_budget(df)

    print(df)

    return api([])


@budget_revenue_bp.route("/vote_revenue", methods=["GET"])
def vote_revenue():
    df = pd.read_csv(csv_url, usecols=[9, 10])
