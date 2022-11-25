from common.api import api
import pandas as pd
import numpy as np
from flask import Blueprint
from common.api import info_csv_url
from handler.rating_handler import clean_rating
from handler.budget_revenue_handler import clean_revenue, clean_budget
from sklearn.ensemble import *
from sklearn.linear_model import *
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.svm import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.2f" % x)

revenue_prediction_bp = Blueprint('revenue_prediction', __name__, url_prefix='/api/revenue_prediction')


@revenue_prediction_bp.route("/data", methods=["GET"])
def api_response():
    """
    :return: api数据
    """
    data_dic = {
        "revenuePredictionModelColumnData": revenue_prediction()[0],
        "revenuePredictionModelLineData": revenue_prediction()[1],
    }
    return api(data_dic)


def revenue_prediction():
    df = pd.read_csv(info_csv_url, usecols=[9, 10, 14])
    df.dropna(inplace=True)
    df = clean_budget(df)
    df = clean_revenue(df)
    df = clean_rating(df)

    data = []
    model_list = ["回归决策树", "线性回归", "Lasso回归", "岭回归", "弹性网络回归", "随机森林回归", "极端随机森林回归", "梯度提升回归",
                  "线性核函数SVM", "高斯核函数SVM"]
    score_list = []
    predict_list = []

    x_train1, x_test1, y_train1, y_test1 = train_test_split(df[["budget", "rating"]], df.revenue, train_size=0.8, random_state=13)
    x_train, x_test, y_train, y_test = x_train1, x_test1, y_train1, y_test1
    # boston = load_boston_model()
    # statistics_graph(boston)
    # linear_progression_train(boston)
    # lasso_model()

    ss = StandardScaler()

    x_train = ss.fit_transform(x_train, y_train)
    x_test = ss.fit_transform(x_test)

    model = DecisionTreeRegressor()
    model.fit(x_train, y_train)
    tree_y_test_predict = model.predict(x_test)
    tree_score = model.score(x_test, y_test)
    tree_avg_score = mean_squared_error(y_test, tree_y_test_predict)
    tree_absolute_score = mean_absolute_error(y_test, tree_y_test_predict)
    score_list.append(tree_score)
    predict_list.append(tree_y_test_predict)
    print(f"回归决策树训练集评价：{model.score(x_train, y_train)}")
    print(f"回归决策树评价：{tree_score}")
    print(f"回归决策树均方误差评价：{tree_avg_score}")
    print(f"回归决策树平均绝对误差评价：{tree_absolute_score}")

    lr = LinearRegression()
    lr.fit(x_train, y_train)
    lr_y_test_predict = lr.predict(x_test)
    lr_score = lr.score(x_test, y_test)
    lr_avg_score = mean_squared_error(y_test, lr_y_test_predict)
    lr_absolute_score = mean_absolute_error(y_test, lr_y_test_predict)
    score_list.append(lr_score)
    predict_list.append(lr_y_test_predict)
    print(f"普通最小二乘线性回归训练集评价：{lr.score(x_train, y_train)}")
    print(f"普通最小二乘线性回归评价：{lr_score}")
    print(f"普通最小二乘线性回归均方误差评价：{lr_avg_score}")
    print(f"普通最小二乘线性回归平均绝对误差评价：{lr_absolute_score}")

    lasso = LassoCV(alphas=np.logspace(-3, 1, 20))
    lasso.fit(x_train, y_train)
    lasso_y_test_predict = lasso.predict(x_test)
    lasso_score = lasso.score(x_test, y_test)
    lasso_avg_score = mean_squared_error(y_test, lasso_y_test_predict)
    lasso_absolute_score = mean_absolute_error(y_test, lasso_y_test_predict)
    score_list.append(lasso_score)
    predict_list.append(lasso_y_test_predict)
    print(f"Lasso L1正则化训练集评价：{lasso.score(x_train, y_train)}")
    print(f"Lasso L1正则化评价：{lasso_score}")
    print(f"Lasso L1正则化均方误差评价：{lasso_avg_score}")
    print(f"Lasso L1正则化平均绝对误差评价：{lasso_absolute_score}")

    ridge = RidgeCV(alphas=np.logspace(-3, 1, 20))
    ridge.fit(x_train, y_train)
    ridge_y_test_predict = ridge.predict(x_test)
    ridge_score = ridge.score(x_test, y_test)
    ridge_avg_score = mean_squared_error(y_test, ridge_y_test_predict)
    ridge_absolute_score = mean_absolute_error(y_test, ridge_y_test_predict)
    score_list.append(ridge_score)
    predict_list.append(ridge_y_test_predict)
    print(f"岭回归 L2正则化训练集评价：{ridge.score(x_train, y_train)}")
    print(f"岭回归 L2正则化评价：{ridge_score}")
    print(f"岭回归 L2正则化均方误差评价：{ridge_avg_score}")
    print(f"岭回归 L2正则化平均绝对误差评价：{ridge_absolute_score}")

    elasticNet = ElasticNetCV(alphas=np.logspace(-3, 1, 20))
    elasticNet.fit(x_train, y_train)
    elasticNet_y_test_predict = elasticNet.predict(x_test)
    elasticNet_score = elasticNet.score(x_test, y_test)
    elasticNet_avg_score = mean_squared_error(y_test, elasticNet_y_test_predict)
    elasticNet_absolute_score = mean_absolute_error(y_test, elasticNet_y_test_predict)
    score_list.append(elasticNet_score)
    predict_list.append(elasticNet_y_test_predict)
    print(f"弹性网络 L1+L2正则化训练集评价：{elasticNet.score(x_train, y_train)}")
    print(f"弹性网络 L1+L2正则化评价：{elasticNet_score}")
    print(f"弹性网络 L1+L2正则化评价均方误差评价：{elasticNet_avg_score}")
    print(f"弹性网络 L1+L2正则化评价平均绝对误差评价：{elasticNet_absolute_score}")

    # 随机森林回归
    rfr = RandomForestRegressor()
    rfr.fit(x_train, y_train)
    rfr_y_test_predict = rfr.predict(x_test)
    rfr_score = rfr.score(x_test, y_test)
    rfr_avg_score = mean_squared_error(y_test, rfr_y_test_predict)
    rfr_absolute_score = mean_absolute_error(y_test, rfr_y_test_predict)
    score_list.append(rfr_score)
    predict_list.append(rfr_y_test_predict)
    print(f"随机森林回归训练集评价：{rfr.score(x_train, y_train)}")
    print(f"随机森林回归评价：{rfr_score}")
    print(f"随机森林回归均方误差评价：{rfr_avg_score}")
    print(f"随机森林回归平均绝对误差评价：{rfr_absolute_score}")

    # 极端随机森林回归
    etr = ExtraTreesRegressor()
    etr.fit(x_train, y_train)
    etr_y_test_predict = etr.predict(x_test)
    etr_score = etr.score(x_test, y_test)
    etr_avg_score = mean_squared_error(y_test, etr_y_test_predict)
    etr_absolute_score = mean_absolute_error(y_test, etr_y_test_predict)
    score_list.append(etr_score)
    predict_list.append(etr_y_test_predict)
    print(f"极端随机森林回归训练集评价：{etr.score(x_train, y_train)}")
    print(f"极端随机森林回归评价：{etr_score}")
    print(f"极端随机森林回归均方误差评价：{etr_avg_score}")
    print(f"极端随机森林回归平均绝对误差评价：{etr_absolute_score}")

    # 梯度提升回归
    gbr = GradientBoostingRegressor()
    gbr.fit(x_train, y_train)
    gbr_y_test_predict = gbr.predict(x_test)
    gbr_score = gbr.score(x_test, y_test)
    gbr_avg_score = mean_squared_error(y_test, gbr_y_test_predict)
    gbr_absolute_score = mean_absolute_error(y_test, gbr_y_test_predict)
    score_list.append(gbr_score)
    predict_list.append(gbr_y_test_predict)
    print(f"梯度提升回归评价：{gbr.score(x_train, y_train)}")
    print(f"梯度提升回归评价：{gbr_score}")
    print(f"梯度提升回归均方误差评价：{gbr_avg_score}")
    print(f"梯度提升回归平均绝对误差评价：{gbr_absolute_score}")

    # 线性核函数SVM预测
    lrsvm = LinearSVR()
    lrsvm.fit(x_train, y_train)
    lrsvm_y_test_predict = lrsvm.predict(x_test)
    lrsvm_score = lrsvm.score(x_test, y_test)
    lrsvm_avg_score = mean_squared_error(y_test, lrsvm_y_test_predict)
    lrsvm_absolute_score = mean_absolute_error(y_test, lrsvm_y_test_predict)
    score_list.append(lrsvm_score)
    predict_list.append(lr_y_test_predict)
    print(f"SVM线性核函数训练集评价：{lrsvm.score(x_train, y_train)}")
    print(f"SVM线性核函数评价：{lrsvm_score}")
    print(f"SVM线性核函数均方误差评价：{lrsvm_avg_score}")
    print(f"SVM线性核函数平均绝对误差评价：{lrsvm_absolute_score}")

    # 高斯核函数SVM预测
    rbfsvm = SVR()
    rbfsvm.fit(x_train, y_train)
    rbfsvm_y_test_predict = lrsvm.predict(x_test)
    rbfsvm_score = lrsvm.score(x_test, y_test)
    rbfsvm_avg_score = mean_squared_error(y_test, rbfsvm_y_test_predict)
    rbfsvm_absolute_score = mean_absolute_error(y_test, rbfsvm_y_test_predict)
    score_list.append(rbfsvm_score)
    predict_list.append(rbfsvm_y_test_predict)
    print(f"SVM高斯核函数训练集评价：{rbfsvm.score(x_train, y_train)}")
    print(f"SVM高斯核函数评价：{rbfsvm_score}")
    print(f"SVM高斯核函数均方误差评价：{rbfsvm_avg_score}")
    print(f"SVM高斯核函数平均绝对误差评价：{rbfsvm_absolute_score}")

    col_data, line_data = [], []
    for s, m in zip(score_list, model_list):
        col_data.append({
            'model': m,
            'score': s
        })

    data.append(col_data)

    for i in range(len(model_list)):
        for j in range(len(x_test)):
            line_data.append({
                'model': model_list[i],
                'x': j,
                'y': predict_list[i][j]
            })
    i = 0
    for j in y_test:
        line_data.append({
            'model': '实际数值',
            'x': i,
            'y': j,
        })
        i += 1

    data.append(line_data)

    return data
