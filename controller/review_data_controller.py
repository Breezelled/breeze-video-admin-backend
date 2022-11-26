from common.api import show_data_api
import pandas as pd
from flask import Blueprint, request
from common.api import obj2json
from model.reviews import Reviews

pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.2f" % x)

review_bp = Blueprint('review', __name__, url_prefix='/api/review')


@review_bp.route("/data", methods=["GET"])
def review_data():
    page, size = int(request.args.get('page')), int(request.args.get('size'))
    reviews = Reviews.query.paginate(page=page, per_page=size)

    return show_data_api(obj2json(reviews.items), reviews.page, reviews.total)
