from common.api import api
import pandas as pd
from flask import Blueprint
from common.api import info_csv_url

pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.2f" % x)

display_bp = Blueprint('display', __name__, url_prefix='/api/display')


@display_bp.route("/data", methods=["GET"])
def api_response():
    """
    :return: api数据
    """
    data_dic = {
        "displayData": display_data(),
    }
    return api(data_dic)


def display_data():
    df = pd.read_csv(info_csv_url)
    df.dropna(inplace=True)
    data = df.to_dict("records")
    print(data)
    # print(df)

    return data

