from flask import jsonify

info_csv_url = "./IMDB/info/info.csv"
review_csv_url = "./IMDB/info/reviews.csv"
freq_words_csv_url = "./IMDB/info/freq_words.csv"


def api(data_dic):
    api = {
        "success": True,
        "data": data_dic,
        "errorMessage": "error message"
    }
    return jsonify(api)


def show_data_api(data, page, total):
    api = {
        "results": data,
        "page": page,
        "total": total,
    }
    return jsonify(api)


def obj2json(items):
    data = []
    for i in items:
        print(i.__str__())
        data.append(i.to_json())

    return data
