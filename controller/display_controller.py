from common.api import show_data_api
import pandas as pd
from flask import Blueprint, request
from common.api import obj2json
from model.info import Info

pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.2f" % x)

display_bp = Blueprint('display', __name__, url_prefix='/api/display')


@display_bp.route("/data", methods=["GET"])
def display_data():
    page, size = int(request.args.get('page')), int(request.args.get('size'))
    info = Info.query.paginate(page=page, per_page=size)

    return show_data_api(obj2json(info.items), info.page, info.total)


# @display_bp.route("/data/search", methods=["GET"])
# def search_data():
#     page, size, val = int(request.args.get('page')), int(request.args.get('size')), \
#                       str(request.args.get('val'))
#     print(val)
#     info = Info.query.filter(Info.name.ilike('%{val}%'.format(val=val))).paginate(page=page, per_page=size)
#
#     return show_data_api(obj2json(info.items), info.page, info.total)
