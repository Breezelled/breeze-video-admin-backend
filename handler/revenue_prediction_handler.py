import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common.api import info_csv_url
from handler.rating_handler import clean_rating
from handler.budget_revenue_handler import clean_revenue, clean_budget
from sklearn.ensemble import *
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import *
from sklearn.model_selection import train_test_split
from sklearn.svm import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import warnings
#
warnings.filterwarnings('ignore')
#
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
#
plt.rcParams["font.sans-serif"] = ["PingFang HK"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
#
# #
# # def statistics_graph(model):
# #     bos = pd.DataFrame(model.data)
# #     bos.columns = model.feature_names
# #     bos['PRICE'] = model.target
# #     bos.head()
# #     X = bos.drop('PRICE', axis=1)
# #     y = bos['PRICE']
#
#
# # def linear_progression_train(boston):
# #     # (2)分割数据
# #     X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.3, random_state=0)
# #     # (3)导入线性回归模型并训练模型
# #     LR = LinearRegression()
# #     LR.fit(X_train, y_train)
# #     # (4)在测试集上预测
# #     y_pred = LR.predict(X_test)
# #     # (5)评估模型
# #     # mse = metrics.mean_squared_error(y_test, y_pred)
# #
# #     plt.scatter(y_test, y_pred)
# #     plt.xlabel("Price: $Y_i$")
# #     plt.ylabel("Predicted prices: $\hat{Y}_i$")
# #     plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
# #     plt.grid()
# #     x = np.arange(0, 50)
# #     y = x
# #     plt.plot(x, y, color='red', lw=4)
# #     plt.text(30, 40, "predict line")
# #     plt.show()
# #     # print("MSE = ", mse)  # 性能评估：模型的均方差
#
#
# # def plot_errors(lambdas, train_errors, test_errors, title):
# #     plt.figure(figsize=(8, 5))
# #     plt.plot(lambdas, train_errors, label="训练集")
# #     plt.plot(lambdas, test_errors, label="测试集")
# #     plt.xlabel("$\\lambda$", fontsize=7)
# #     plt.ylabel("MSE", fontsize=7)
# #     plt.title(title, fontsize=14)
# #     plt.legend(fontsize=12)
# #     plt.show()
# #
# #
# # def evaluate_model(Model, lambdas):
# #     training_errors = []
# #     testing_errors = []
# #     for l in lambdas:
# #         model = Model(alpha=1, max_iter=1000)
# #         model.fit(X_train, y_train)
# #
# #         training_predictions = model.predict(X_train)
# #         # training_mse = metrics.mean_squared_error(y_train, training_predictions)
# #         # training_errors.append(training_mse)
# #         # print(training_mse)
# #
# #         testing_predictions = model.predict(X_test)
# #         # testing_mse = metrics.mean_squared_error(y_test, testing_predictions)
# #         # testing_errors.append(testing_mse)
# #
# #     return training_errors, testing_errors
# #
# #
# # def lasso_model():
# #     lambdas = np.arange(0, 10, step=0.1)
# #     lasso_train, lasso_test = evaluate_model(Lasso, lambdas)
# #     print(lasso_train)
# #     plot_errors(lambdas, lasso_train, lasso_test, "Lasso/L1")
# #     lambdas = np.arange(0, 10, step=0.1)
# #     ridge_train, ridge_test = evaluate_model(Ridge, lambdas)
# #     print(ridge_train)
# #     plot_errors(lambdas, ridge_train, ridge_test, "Ridge/L2")
# #     lambdas = np.arange(0, 10, step=0.1)
# #     elasticNet_train, elasticNet_test = evaluate_model(ElasticNet, lambdas)
# #     print(elasticNet_train)
# #     plot_errors(lambdas, elasticNet_train, elasticNet_test, "ElasticNet/L1+L2")
#
#
if __name__ == "__main__":
    df = pd.read_csv("." + info_csv_url, usecols=[9, 10, 14])
    df.dropna(inplace=True)
    df = clean_budget(df)
    df = clean_revenue(df)
    df = clean_rating(df)
    # boston = load_boston()
    # print(boston)

    x_train1, x_test1, y_train1, y_test1 = train_test_split(df[["rating", "budget"]], df.revenue, train_size=0.8, random_state=13)
    X_train, X_test, y_train, y_test = x_train1, x_test1, y_train1, y_test1
    # boston = load_boston_model()
    # statistics_graph(boston)
    # linear_progression_train(boston)
    # lasso_model()

    ss = StandardScaler()

    X_train = ss.fit_transform(X_train, y_train)
    X_test = ss.fit_transform(X_test)

    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    tree_y_test_predict = model.predict(X_test)
    tree_score = model.score(X_test, y_test)
    # tree_avg_score = mean_squared_error(y_test, tree_y_test_predict)
    tree_absolute_score = mean_absolute_error(y_test, tree_y_test_predict)
    print(f"回归树训练集评价：{model.score(X_train, y_train)}")
    print(f"回归树评价：{tree_score}")
    # print(f"回归树均方误差评价：{tree_avg_score}")
    print(f"回归树平均绝对误差评价：{tree_absolute_score}")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_y_test_predict = lr.predict(X_test)
    lr_score = lr.score(X_test, y_test)
    # lr_avg_score = mean_squared_error(y_test, lr_y_test_predict)
    lr_absolute_score = mean_absolute_error(y_test, lr_y_test_predict)
    print(f"普通最小二乘线性回归训练集评价：{lr.score(X_train, y_train)}")
    print(f"普通最小二乘线性回归评价：{lr_score}")
    # print(f"普通最小二乘线性回归均方误差评价：{lr_avg_score}")
    print(f"普通最小二乘线性回归平均绝对误差评价：{lr_absolute_score}")

    lasso = LassoCV(alphas=np.logspace(-3, 1, 20))
    lasso.fit(X_train, y_train)
    lasso_y_test_predict = lasso.predict(X_test)
    lasso_score = lasso.score(X_test, y_test)
    # lasso_avg_score = mean_squared_error(y_test, lasso_y_test_predict)
    lasso_absolute_score = mean_absolute_error(y_test, lasso_y_test_predict)
    print(f"Lasso L1正则化训练集评价：{lasso.score(X_test, y_test)}")
    print(f"Lasso L1正则化评价：{lasso_score}")
    # print(f"Lasso L1正则化均方误差评价：{lasso_avg_score}")
    print(f"Lasso L1正则化平均绝对误差评价：{lasso_absolute_score}")

    ridge = RidgeCV(alphas=np.logspace(-3, 1, 20))
    ridge.fit(X_train, y_train)
    ridge_y_test_predict = ridge.predict(X_test)
    ridge_score = ridge.score(X_test, y_test)
    # ridge_avg_score = mean_squared_error(y_test, ridge_y_test_predict)
    ridge_absolute_score = mean_absolute_error(y_test, ridge_y_test_predict)
    print(f"岭回归 L2正则化评价：{ridge_score}")
    # print(f"岭回归 L2正则化均方误差评价：{ridge_avg_score}")
    print(f"岭回归 L2正则化平均绝对误差评价：{ridge_absolute_score}")

    elasticNet = ElasticNetCV(alphas=np.logspace(-3, 1, 20))
    elasticNet.fit(X_train, y_train)
    elasticNet_y_test_predict = elasticNet.predict(X_test)
    elasticNet_score = elasticNet.score(X_test, y_test)
    # elasticNet_avg_score = mean_squared_error(y_test, elasticNet_y_test_predict)
    elasticNet_absolute_score = mean_absolute_error(y_test, elasticNet_y_test_predict)
    print(f"L1+L2正则化评价：{elasticNet_score}")
    # print(f"L1+L2正则化评价均方误差评价：{elasticNet_avg_score}")
    print(f"L1+L2正则化评价平均绝对误差评价：{elasticNet_absolute_score}")

    # 随机森林回归
    rfr = RandomForestRegressor()
    rfr.fit(X_train, y_train)
    rfr_y_test_predict = rfr.predict(X_test)
    rfr_score = rfr.score(X_test, y_test)
    # rfr_avg_score = mean_squared_error(y_test, rfr_y_test_predict)
    rfr_absolute_score = mean_absolute_error(y_test, rfr_y_test_predict)
    print(f"随机森林回归：{rfr_score}")
    # print(f"随机森林回归均方误差评价：{rfr_avg_score}")
    print(f"随机森林回归平均绝对误差评价：{rfr_absolute_score}")

    # 极端随机森林回归
    etr = ExtraTreesRegressor()
    etr.fit(X_train, y_train)
    etr_y_test_predict = etr.predict(X_test)
    etr_score = etr.score(X_test, y_test)
    # etr_avg_score = mean_squared_error(y_test, etr_y_test_predict)
    etr_absolute_score = mean_absolute_error(y_test, etr_y_test_predict)
    print(f"极端随机森林回归：{etr_score}")
    # print(f"极端随机森林回归均方误差评价：{etr_avg_score}")
    print(f"极端随机森林回归平均绝对误差评价：{etr_absolute_score}")

    # 梯度提升回归
    gbr = GradientBoostingRegressor()
    gbr.fit(X_train, y_train)
    gbr_y_test_predict = gbr.predict(X_test)
    gbr_score = gbr.score(X_test, y_test)
    # gbr_avg_score = mean_squared_error(y_test, gbr_y_test_predict)
    gbr_absolute_score = mean_absolute_error(y_test, gbr_y_test_predict)
    print(f"梯度提升回归：{gbr_score}")
    # print(f"梯度提升回归均方误差评价：{gbr_avg_score}")
    print(f"梯度提升回归平均绝对误差评价：{gbr_absolute_score}")

    # 线性核函数SVM预测
    lrsvm = LinearSVR()
    lrsvm.fit(X_train, y_train)
    lrsvm_y_test_predict = lrsvm.predict(X_test)
    lrsvm_score = lrsvm.score(X_test, y_test)
    # lrsvm_avg_score = mean_squared_error(y_test, lrsvm_y_test_predict)
    lrsvm_absolute_score = mean_absolute_error(y_test, lrsvm_y_test_predict)
    print(f"SVM线性核函数：{gbr_score}")
    # print(f"SVM线性核函数均方误差评价：{lrsvm_avg_score}")
    print(f"SVM线性核函数平均绝对误差评价：{lrsvm_absolute_score}")

    # 高斯核函数SVM预测
    lrsvm = SVR()
    lrsvm.fit(X_train, y_train)
    lrsvm_y_test_predict = lrsvm.predict(X_test)
    lrsvm_score = lrsvm.score(X_test, y_test)
    lrsvm_avg_score = mean_squared_error(y_test, lrsvm_y_test_predict)
    lrsvm_absolute_score = mean_absolute_error(y_test, lrsvm_y_test_predict)
    print(f"SVM线性核函数：{gbr_score}")
    print(f"SVM线性核函数均方误差评价：{lrsvm_avg_score}")
    print(f"SVM线性核函数平均绝对误差评价：{lrsvm_absolute_score}")

    plt.figure(figsize=(24, 12), facecolor='w')
    ln_x_test = range(len(X_test))
    print(ln_x_test, lr_y_test_predict)
    print(ln_x_test, len(tree_y_test_predict))

    plt.plot(ln_x_test, y_test, 'r-', lw=2, label='实际数值')
    plt.plot(ln_x_test, lr_y_test_predict, 'b-', lw=2, label='线性回归$R^2$=%.3f' % lr_score)
    plt.plot(ln_x_test, tree_y_test_predict, 'd-', lw=4, label='回归决策树$R^2$=%.3f' % tree_score)
    plt.plot(ln_x_test, lasso_y_test_predict, 'y-', lw=2, label='Lasso/L1回归$R^2$=%.3f' % lasso_score)
    plt.plot(ln_x_test, ridge_y_test_predict, 'c-', lw=2, label='岭回归/L2回归$R^2$=%.3f' % ridge_score)
    plt.plot(ln_x_test, elasticNet_y_test_predict, 'd-', lw=2, label='弹性网络/L1+L2回归$R^2$=%.3f' % elasticNet_score)
    plt.plot(ln_x_test, elasticNet_y_test_predict, 'd-', lw=2, label='随机森林回归$R^2$=%.3f' % rfr_score)
    plt.plot(ln_x_test, elasticNet_y_test_predict, 'd-', lw=2, label='极端随机森林回归$R^2$=%.3f' % etr_score)
    plt.plot(ln_x_test, elasticNet_y_test_predict, 'd-', lw=2, label='梯度提升回归$R^2$=%.3f' % gbr_score)
    plt.plot(ln_x_test, elasticNet_y_test_predict, 'd-', lw=2, label='SVM线性核函数$R^2$=%.3f' % lrsvm_score)
    plt.xlabel('测试集数量', fontsize=18)
    plt.ylabel('票房', fontsize=18)
    plt.legend(loc='upper center')
    plt.grid(True)
    plt.title('票房预测R方评价', fontsize=20)
    plt.show()

    # plt.figure(figsize=(24, 12), facecolor='w')
    # plt.plot(ln_x_test, y_test, 'r-', lw=2, label='实际数值')
    # plt.plot(ln_x_test, lr_y_test_predict, 'b-', lw=2, label='线性回归$R^2$=%.3f' % lr_avg_score)
    # plt.plot(ln_x_test, tree_y_test_predict, 'd-', lw=4, label='回归决策树$R^2$=%.3f' % tree_avg_score)
    # plt.plot(ln_x_test, lasso_y_test_predict, 'y-', lw=2, label='Lasso/L1回归$R^2$=%.3f' % lasso_avg_score)
    # plt.plot(ln_x_test, ridge_y_test_predict, 'c-', lw=2, label='岭回归/L2回归$R^2$=%.3f' % ridge_avg_score)
    # plt.plot(ln_x_test, elasticNet_y_test_predict, 'd-', lw=2, label='弹性网络/L1+L2回归$R^2$=%.3f' % elasticNet_avg_score)
    # plt.plot(ln_x_test, elasticNet_y_test_predict, 'd-', lw=2, label='随机森林回归$R^2$=%.3f' % rfr_avg_score)
    # plt.plot(ln_x_test, elasticNet_y_test_predict, 'd-', lw=2, label='极端随机森林回归$R^2$=%.3f' % etr_avg_score)
    # plt.plot(ln_x_test, elasticNet_y_test_predict, 'd-', lw=2, label='梯度提升回归$R^2$=%.3f' % gbr_avg_score)
    # plt.xlabel('测试集数量', fontsize=18)
    # plt.ylabel('房价', fontsize=18)
    # plt.legend(loc='upper center')
    # plt.grid(True)
    # plt.title('Boston房价预测均方误差评价', fontsize=20)
    # plt.show()

    plt.figure(figsize=(24, 12), facecolor='w')
    plt.plot(ln_x_test, y_test, 'r-', lw=2, label='实际数值')
    plt.plot(ln_x_test, lr_y_test_predict, 'b-', lw=2, label='线性回归$R^2$=%.3f' % lr_absolute_score)
    plt.plot(ln_x_test, tree_y_test_predict, 'd-', lw=4, label='回归决策树$R^2$=%.3f' % tree_absolute_score)
    plt.plot(ln_x_test, lasso_y_test_predict, 'y-', lw=2, label='Lasso/L1回归$R^2$=%.3f' % lasso_absolute_score)
    plt.plot(ln_x_test, ridge_y_test_predict, 'c-', lw=2, label='岭回归/L2回归$R^2$=%.3f' % ridge_absolute_score)
    plt.plot(ln_x_test, elasticNet_y_test_predict, 'd-', lw=2,
             label='弹性网络/L1+L2回归$R^2$=%.3f' % elasticNet_absolute_score)
    plt.plot(ln_x_test, elasticNet_y_test_predict, 'd-', lw=2, label='随机森林回归$R^2$=%.3f' % rfr_absolute_score)
    plt.plot(ln_x_test, elasticNet_y_test_predict, 'd-', lw=2, label='极端随机森林回归$R^2$=%.3f' % etr_absolute_score)
    plt.plot(ln_x_test, elasticNet_y_test_predict, 'd-', lw=2, label='梯度提升回归$R^2$=%.3f' % gbr_absolute_score)
    plt.xlabel('测试集数量', fontsize=18)
    plt.ylabel('票房', fontsize=18)
    plt.legend(loc='upper center')
    plt.grid(True)
    plt.title('票房预测平均绝对误差评价', fontsize=20)
    plt.show()

    # 参数优化
    pipes = [
        Pipeline([
            ('mms', MinMaxScaler()),
            ('pca', PCA()),
            ('decision', DecisionTreeRegressor(criterion='squared_error'))
        ]),
        Pipeline([
            ('mms', MinMaxScaler()),
            ('decision', DecisionTreeRegressor(criterion='squared_error'))
        ]),
        Pipeline([
            ('decision', DecisionTreeRegressor(criterion='squared_error'))
        ]),
    ]

    # 参数
    parameters = [
        {
            "pca__n_components": [0.25, 0.5, 0.75, 1],
            "decision__max_depth": np.linspace(1, 20, 20).astype(np.int8)
        },
        {
            "decision__max_depth": np.linspace(1, 20, 20).astype(np.int8)
        },
        {
            "decision__max_depth": np.linspace(1, 20, 20).astype(np.int8)
        },
    ]
    x_train2, x_test2, y_train2, y_test2 = x_train1, x_test1, y_train1, y_test1
    # for index in range(3):
        # pipe = pipes[index]
        # gscv = GridSearchCV(pipe, param_grid=parameters[index])
        # gscv.fit(x_train2, y_train2)
        # print(index, f"评价值：{gscv.best_score_}", f"最优参数为：{gscv.best_params_}")

mms_best = MinMaxScaler()
decision3 = DecisionTreeRegressor()

x_train3, x_test3, y_train3, y_test3 = x_train1, x_test1, y_train1, y_test1

x_train3 = mms_best.fit_transform(x_train3, y_train3)
x_test3 = mms_best.transform(x_test3)
decision3.fit(x_train3, y_train3)

print("深度为5——最优参数看正确率", decision3.score(x_test3, y_test3))

x_train4, x_test4, y_train4, y_test4 = x_train1, x_test1, y_train1, y_test1

depths = np.arange(1, 20)
err_list = []
for depth in depths:
    clf = DecisionTreeRegressor()
    clf.fit(x_train4, y_train4)

    score1 = clf.score(x_test4, y_test4)
    err = 1 - score1
    err_list.append(err)
    print(f"深度为：{depth},正确率为%.5f" % score1)

plt.figure(facecolor='w')
plt.plot(depths, err_list, 'ro-', lw=3)
plt.xlabel("决策树深度", fontsize=16)
plt.ylabel("错误率", fontsize=16)
plt.grid(True)
plt.title("决策树层数与拟合问题相关性", fontsize=20)
plt.show()

if __name__ == "__main__":
    data = pd.read_csv("."+info_csv_url)
    data.dropna(inplace=True)
    predict_feature = "revenue"
    features = ["rating", "budget"]
