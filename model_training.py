import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from xgboost.sklearn import XGBRegressor
from lightgbm.sklearn import LGBMRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.neural_network import MLPRegressor as MLP
from sklearn.metrics import mean_squared_error, median_absolute_error, mean_squared_log_error, mean_absolute_error, \
    r2_score
import global_config as cfg


def train_model(X, y, model_type=None, param_name=None, cv_times=10) -> dict:
    if model_type is None or model_type not in cfg.model_enum:
        raise ValueError
    params = cfg.model_params.get(model_type)
    # 获取模型类型及其参数并初始化
    if model_type == cfg.model_enum[0]:  # XGB
        model = XGBRegressor(**params.get(param_name))
    elif model_type == cfg.model_enum[1]:  # LGBM
        model = LGBMRegressor(**params.get(param_name))
    elif model_type == cfg.model_enum[2]:  # SVM
        model = SVR(**params.get(param_name))
    elif model_type == cfg.model_enum[3]:  # RF
        model = RF(**params.get(param_name))
    elif model_type == cfg.model_enum[4]:  # MLP
        model = MLP(**params.get(param_name))

    cv = KFold(n_splits=cv_times, shuffle=True)

    r2_scores = np.empty(cv_times)
    rmse_scores = np.empty(cv_times)
    nrmse_scores = np.empty(cv_times)
    print(f"Model type: {model_type}, param name: {param_name}")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    for idx, (train_idx, test_idx) in enumerate(cv.split(X_train, y_train)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if model_type == cfg.model_enum[0]:  # XGB
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=False)
        elif model_type == cfg.model_enum[1]:  # LGBM
            callbacks = [lgb.log_evaluation(period=0)]
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=callbacks)
        # elif model_type == cfg.model_enum[2]:   #SVM
        #     model.fit(X_train, y_train)
        # elif model_type == cfg.model_enum[3]:   #RF
        #     # print(X_train)
        #     model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)
        preds = model.predict(X_test)

        r2 = r2_score(y_test, preds)
        r2_scores[idx] = r2

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        rmse_scores[idx] = rmse

        nrmse = rmse / (y_test.max() - y_test.min())
        nrmse_scores[idx] = nrmse

    preds = model.predict(X_val)
    val_r2 = r2_score(y_val, preds)
    val_rmse = np.sqrt(mean_squared_error(y_val, preds))
    val_nrmse = val_rmse / (y_val.max() - y_val.min())
    # print("Validation r2: ", val_r2)
    # print("Validation rmse: ", val_rmse)
    res = {"R2": r2_scores,
           "RMSE": rmse_scores,
           "NRMSE": nrmse_scores,
           "Val_R2": val_r2,
           "Val_RMSE": val_rmse,
           "Val_NRMSE": val_nrmse}
    return res


def train_ratio_model(X, y, model_type=None, cv_times=5):
    if model_type is None:
        raise ValueError
    # 获取模型类型及其参数并初始化
    if model_type == cfg.model_enum[0]:  # XGB
        params = {
            'n_estimators': 2300,
            'learning_rate': 0.008,
            'max_depth': 22,
            'lambda': 0.8777358996534239,
            'alpha': 0.02495760060129463,
            'min_child_weight': 12,
            'gamma': 20,
            'colsample_bytree': 0.1,
            'colsample_bylevel': 0.4,
            'colsample_bynode': 0.5,
        }
        model = XGBRegressor(**params)
    elif model_type == cfg.model_enum[1]:  # LGBM
        params = None
        model = LGBMRegressor(**params)
    elif model_type == cfg.model_enum[2]:  # SVM
        params = None
        model = SVR(**params)
    elif model_type == cfg.model_enum[3]:  # RF
        params = None
        model = RF(**params)

    cv = KFold(n_splits=cv_times, shuffle=True)

    r2_scores = np.empty(cv_times)
    rmse_scores = np.empty(cv_times)
    print(model_type)

    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if model_type == cfg.model_enum[0]:  # XGB
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=False)
        elif model_type == cfg.model_enum[1]:  # LGBM
            callbacks = [lgb.log_evaluation(period=0)]
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=callbacks)
        elif model_type == cfg.model_enum[2]:  # SVM

            model.fit(X_train, y_train)
        elif model_type == cfg.model_enum[3]:  # RF
            # print(X_train)
            model.fit(X_train, y_train)

        preds = model.predict(X_test)

        r2 = r2_score(y_test, preds)
        r2_scores[idx] = r2

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        rmse_scores[idx] = rmse

    return r2_scores, rmse_scores
