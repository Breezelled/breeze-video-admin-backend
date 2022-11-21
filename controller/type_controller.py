from common.api import api
import pandas as pd
from flask import Blueprint
from handler.budget_revenue_handler import clean_revenue, clean_budget_revenue_exception, clean_budget
from handler.type_handler import clean_type_numeric, clean_type
from common.api import csv_url

pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.2f" % x)

type_bp = Blueprint('type', __name__, url_prefix='/api/type')


@type_bp.route("/data", methods=["GET"])
def api_response():
    data_dic = {
        "typeBudgetData": type_budget(),
        "typeRevenueData": type_revenue(),
        "typeDateNumData": type_date_num(),
    }
    return api(data_dic)


def type_budget():
    df = pd.read_csv(csv_url, usecols=[3, 9])
    df.dropna(inplace=True)
    df = clean_budget(df)

    data = clean_type_numeric(df, 'budget').to_dict("records")

    print(data)
    # print(df)

    return data


def type_revenue():
    df = pd.read_csv(csv_url, usecols=[3, 10])
    df.dropna(inplace=True)
    df = clean_revenue(df)

    data = clean_type_numeric(df, 'revenue').to_dict("records")

    print(data)
    # print(df)

    return data


def type_date_num():
    df = pd.read_csv(csv_url, usecols=[3, 4])
    df.dropna(inplace=True)
    df = clean_type(df)
    df['release_date'] = df['release_date'].str.extract(r'(\d\d\d\d)', expand=False)
    df['cnt'] = 1
    data = df.groupby(['type', 'release_date'])['cnt'].size()\
        .reset_index(name='count')
    data = data.sort_values(by='release_date', ascending=True).to_dict("records")
    # data = data.reset_index()
    print(data)
    # print(df)

    return data
