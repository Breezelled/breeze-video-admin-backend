from common.api import api
import pandas as pd
from flask import Blueprint
from handler.budget_revenue_handler import clean_revenue, clean_budget_revenue_exception, \
    clean_budget
from handler.release_date_handler import clean_date_to_year
from handler.company_handler import clean_company
from common.api import info_csv_url

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
        "companyBudgetRevenueData": company_budget_revenue(),
    }
    return api(data_dic)


def budget_revenue():
    df = pd.read_csv(info_csv_url, usecols=[9, 10])
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
    df = pd.read_csv(info_csv_url, usecols=[4, 9])
    df.dropna(inplace=True)
    df = clean_budget(df)
    df = clean_date_to_year(df)
    data = df.groupby(df['release_date']).mean().round(2).reset_index().to_dict("records")

    return data


def revenue():
    df = pd.read_csv(info_csv_url, usecols=[4, 10])
    df.dropna(inplace=True)
    df = clean_revenue(df)
    df = clean_date_to_year(df)
    data = df.groupby(df['release_date']).mean().round(2).reset_index().to_dict("records")

    return data


def company_budget_revenue():
    df_r = pd.read_csv(info_csv_url, usecols=[10, 12])
    df_b = pd.read_csv(info_csv_url, usecols=[9, 12])
    df_r.dropna(inplace=True)
    df_b.dropna(inplace=True)
    df_b = clean_budget(df_b)
    df_r = clean_revenue(df_r)
    df_b = clean_company(df_b)
    df_r = clean_company(df_r)
    df_b = df_b.groupby(df_b['company']).mean().round(2).reset_index()
    df_r = df_r.groupby(df_r['company']).mean().round(2).reset_index()
    data = pd.merge(df_b, df_r, on='company').to_dict("records")
    print(data)

    return data
