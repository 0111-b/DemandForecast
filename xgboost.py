import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_datapath = './input/demand-forecasting-kernels-only/train.csv'
test_datapath = './input/demand-forecasting-kernels-only/test.csv'
submission_datapath = './input/demand-forecasting-kernels-only/sample_submission.csv'

df_train = pd.read_csv(train_datapath)
df_test = pd.read_csv(test_datapath)
df_submission = pd.read_csv(submission_datapath)

def convert_dates(x):
    x['date']=pd.to_datetime(x['date'])
    x['month']=x['date'].dt.month
    x['year']=x['date'].dt.year
    x['dayofweek']=x['date'].dt.dayofweek
    x.pop('date')
    return x

def add_avg(x):
    x['daily_avg']=x.groupby(['item','store','dayofweek'])['sales'].transform('mean') #daily_avg column based on sales per day
    x['monthly_avg']=x.groupby(['item','store','month'])['sales'].transform('mean') #monthly_avg column based on sales per month
    return x

def merge(x,y,col,col_name):
    x =pd.merge(x, y, how='left', on=None, left_on=col, right_on=col,
            left_index=False, right_index=False, sort=True,
             copy=True, indicator=False,validate=None)
    x=x.rename(columns={'sales':col_name})
    return x

def XGBmodel(x_train,x_test,y_train,y_test):
    matrix_train = xgb.DMatrix(x_train,label=y_train)
    matrix_test = xgb.DMatrix(x_test,label=y_test)
    model=xgb.train(params={'objective':'reg:linear','eval_metric':'mae'} #reg:linear cuz target value is a regression, mae for mean absolute error, can be rmse as well. More info - see documentation
                    ,dtrain=matrix_train,num_boost_round=200,
                    early_stopping_rounds=20,evals=[(matrix_test,'test')],) #early_stopping_rounds = 20 : stop if 20 consequent rounds without decrease of error
    return model

if __name__ == '__main__':
    df_train = convert_dates(df_train)
    df_test = convert_dates(df_test)

    df_train = add_avg(df_train)

    daily_avg = df_train.groupby(['item', 'store', 'dayofweek'])['sales'].mean().reset_index()
    monthly_avg = df_train.groupby(['item', 'store', 'month'])['sales'].mean().reset_index()

    df_test = merge(df_test, daily_avg, ['item', 'store', 'dayofweek'], 'daily_avg')
    df_test = merge(df_test, monthly_avg, ['item', 'store', 'month'], 'monthly_avg')

    df_test.sample(10)

    x_train, x_test, y_train, y_test = train_test_split(df_train.drop('sales', axis=1), df_train.pop('sales'),
                                                        random_state=123,
                                                        test_size=0.2)  # 将数据集分为train和test
    # xgboost
    model = XGBmodel(x_train, x_test, y_train, y_test)

    x_test_pred = model.predict(xgb.DMatrix(x_test))

    mean_squared_error(y_true=y_test,
                       y_pred=x_test_pred)

    root_mean_sqaure_error_RMSE = sqrt(mean_squared_error(y_true=y_test, y_pred=x_test_pred))
    print(root_mean_sqaure_error_RMSE)

    mean_absolute_error(y_true=y_test,
                        y_pred=x_test_pred)
    submission = pd.DataFrame(df_test.pop('id'))
    print(submission.head())

    y_pred = model.predict(xgb.DMatrix(df_test), ntree_limit=model.best_ntree_limit)
    submission['sales'] = y_pred
    submission.to_csv('submission.csv', index=False)
