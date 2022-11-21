from flask import jsonify

csv_url = "./IMDB/info/info.csv"


def api(data_dic):
    api = {
        "success": True,
        "data": data_dic,
        "errorMessage": "error message"
    }
    return jsonify(api)
