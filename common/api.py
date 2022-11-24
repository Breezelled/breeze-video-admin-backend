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
