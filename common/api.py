from flask import jsonify


def api(data_dic):

    api = {
        "success": True,
        "data": data_dic,
        "errorMessage": "error message"
    }
    return jsonify(api)
