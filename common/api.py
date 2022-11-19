from flask import jsonify


def api(data):
    api = {
        "success": True,
        "data": {"budgetRevenueData": data},
        "errorMessage": "error message"
    }
    return jsonify(api)
