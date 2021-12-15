# Import necessary libraries and make necessary arrangements
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
import warnings
import pandas_profiling as pp
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.filterwarnings('ignore')

# 用来研究data的函数
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

# 根据date创建特征
def create_date_features(df):
    df['month'] = df.date.dt.month # 对应年份的月份
    df['day_of_month'] = df.date.dt.day # 对应月份的哪天
    df['day_of_year'] = df.date.dt.dayofyear # 相应年份的哪一天
    df['week_of_year'] = df.date.dt.weekofyear # 相应年份的哪个星期
    df['day_of_week'] = df.date.dt.dayofweek # 每个月对应星期的哪一天
    df['year'] = df.date.dt.year # 哪年
    df["is_wknd"] = df.date.dt.weekday // 4 # 从0开始，0-4表示周一到周五，5,6代表周末， 周末//4得到1，因此可以代表这天是否是周末
    df['is_month_start'] = df.date.dt.is_month_start.astype(int) # 是否是月份的开始
    df['is_month_end'] = df.date.dt.is_month_end.astype(int) # 是否是月份的结束
    return df

# 随机噪声
def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),)) # 高斯随机噪声

# Lag/Shifted Features
def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["store", "item"])['sales'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe

# 滚动平均特征
def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["store", "item"])['sales']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe

# 指数加权平均特征
def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe


def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val


def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False

# 特征重要程度
def plot_lgb_importances(model, plot=False, num=10):

    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))

# Kaggle input part
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

if __name__ == '__main__':
    # 加载数据集
    train = pd.read_csv('./input/demand-forecasting-kernels-only/train.csv', parse_dates=['date']) # 格式转换
    test = pd.read_csv('./input/demand-forecasting-kernels-only/test.csv', parse_dates=['date'])
    sample_sub = pd.read_csv('./input/demand-forecasting-kernels-only/sample_submission.csv')
    df = pd.concat([train, test], sort=False)

    # 查看数据特征
    print("-----------------------train.csv-----------------------")
    train.info()
    check_df(train)
    print("-----------------------test.csv-----------------------")
    check_df(test)
    print("-----------------------sample_submission.csv-----------------------")
    check_df(sample_sub)

    # 数据总览分析
    # report = pp.ProfileReport(train)
    # report.to_file("report-train.html")
    # report = pp.ProfileReport(test)
    # report.to_file("report-test.html")

    result = pd.read_csv('submission.csv')
    report = pp.ProfileReport(result)
    report.to_file("submission.html")

    # 特征分析
    # df["sales"].describe([0.10, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99])
    # df[["store"]].nunique()
    # df[["item"]].nunique()
    # df.groupby(["store"])["item"].nunique()
    # df.groupby(["store", "item"]).agg({"sales": ["sum"]})
    # df.groupby(["store", "item"]).agg({"sales": ["sum", "mean", "median", "std"]})
    # df = create_date_features(df)
    # check_df(df)
    # df.groupby(["store", "item", "month"]).agg({"sales": ["sum", "mean", "median", "std"]})
    # df.sort_values(by=['store', 'item', 'date'], axis=0, inplace=True)
    # df = lag_features(df, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728])
    # check_df(df)
    # df = roll_mean_features(df, [365, 546])
    # df.tail()
    # alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
    # lags = [91, 98, 105, 112, 180, 270, 365, 546, 728]
    #
    # df = ewm_features(df, alphas, lags)
    #
    # check_df(df)
    # df = pd.get_dummies(df, columns=['store', 'item', 'day_of_week', 'month'])
    # df['sales'] = np.log1p(df["sales"].values)
    # train = df.loc[(df["date"] < "2017-01-01"), :]
    #
    # # Validation set including first 3 months of 2017 (as we will forecast the first 3 months of 2018)
    # val = df.loc[(df["date"] >= "2017-01-01") & (df["date"] < "2017-04-01"), :]
    # cols = [col for col in train.columns if col not in ['date', 'id', "sales", "year"]]
    # Y_train = train['sales']
    # X_train = train[cols]
    #
    # Y_val = val['sales']
    # X_val = val[cols]
    # Y_train.shape, X_train.shape, Y_val.shape, X_val.shape
    # lgb_params = {'metric': {'mae'},
    #               'num_leaves': 10,
    #               'learning_rate': 0.02,
    #               'feature_fraction': 0.8,
    #               'max_depth': 5,
    #               'verbose': 0,
    #               'num_boost_round': 1000,
    #               'early_stopping_rounds': 200,
    #               'nthread': -1}
    # lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
    # lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)
    # model = lgb.train(lgb_params, lgbtrain,
    #                   valid_sets=[lgbtrain, lgbval],
    #                   num_boost_round=lgb_params['num_boost_round'],
    #                   early_stopping_rounds=lgb_params['early_stopping_rounds'],
    #                   feval=lgbm_smape,
    #                   verbose_eval=100)
    # y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)
    # smape(np.expm1(y_pred_val), np.expm1(Y_val))
    # plot_lgb_importances(model, num=30, plot=True)
    # train = df.loc[~df.sales.isna()]
    # Y_train = train['sales']
    # X_train = train[cols]
    #
    # test = df.loc[df.sales.isna()]
    # X_test = test[cols]
    # lgb_params = {'metric': {'mae'},
    #               'num_leaves': 10,
    #               'learning_rate': 0.02,
    #               'feature_fraction': 0.8,
    #               'max_depth': 5,
    #               'verbose': 0,
    #               'nthread': -1,
    #               "num_boost_round": model.best_iteration}
    # lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
    # model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)
    # test_preds = model.predict(X_test, num_iteration=model.best_iteration)
    #
    # submission_df = test.loc[:, ['id', 'sales']]
    # submission_df['sales'] = np.expm1(test_preds)
    # submission_df['id'] = submission_df.id.astype(int)
    #
    # submission_df.to_csv('submission.csv', index=False)
    # submission_df.head(20)