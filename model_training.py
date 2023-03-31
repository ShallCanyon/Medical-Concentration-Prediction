import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearnex import patch_sklearn, unpatch_sklearn
unpatch_sklearn()
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from xgboost.sklearn import XGBRegressor
from lightgbm.sklearn import LGBMRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.neural_network import MLPRegressor as MLP
from DataLogger import DataLogger
from sklearn.metrics import mean_squared_error, median_absolute_error, mean_squared_log_error, mean_absolute_error, \
    r2_score
import global_config as cfg

logger = cfg.logger


def train_model(X, y, model_type=None, params=None, cv_times=10):
    if model_type is None or model_type not in cfg.model_enum:
        raise ValueError
    if params is None:
        params = dict()
    # params = cfg.model_params.get(model_type)
    # 获取模型类型及其参数并初始化
    if model_type == cfg.model_enum[0]:  # XGB
        # model = XGBRegressor(**params.get(params))
        model = XGBRegressor(**params)
    elif model_type == cfg.model_enum[1]:  # LGBM
        # model = LGBMRegressor(**params.get(params))
        model = LGBMRegressor(**params)
    elif model_type == cfg.model_enum[2]:  # SVM
        # model = SVR(**params.get(params))
        model = SVR(**params)
    elif model_type == cfg.model_enum[3]:  # RF
        # model = RF(**params.get(params))
        model = RF(**params)
    elif model_type == cfg.model_enum[4]:  # MLP
        # model = MLP(**params.get(params))
        model = MLP(**params)
    elif model_type == cfg.model_enum[-1]:  # Custom
        from sklearn.linear_model import LinearRegression
        from sklearn.linear_model import SGDRegressor
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.linear_model import Ridge
        from sklearn.linear_model import Lasso
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.tree import ExtraTreeRegressor
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.ensemble import BaggingRegressor
        model = BaggingRegressor()

    cv = KFold(n_splits=cv_times, shuffle=True)

    r2_scores = np.empty(cv_times)
    rmse_scores = np.empty(cv_times)
    nrmse_scores = np.empty(cv_times)
    # print(f"Model type: {model_type}, parameter: {params}")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

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
    return res, model


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


def model_training(X, y, model_type, model_params=None, train_times=1):
    """
    使用指定的模型进行模型训练，训练结果由日志打印
    :param X: 传入的数据特征
    :param y: 传入的数据标签
    :param model_type: 指定的模型名
    :param model_params: 模型的参数，默认为空
    :param train_times: 训练的模型数
    :return 返回训练完成的模型
    """
    if train_times == 1:
        result, model = train_model(X, y, model_type=model_type, params=model_params)
        logger.info("10 fold validation: ")
        logger.info("\tR2 Scores: %0.3f (+/- %0.3f)" %
                    (result.get("R2").mean(), result.get("R2").std()))
        logger.info("\tRMSE Scores: %0.3f (+/- %0.3f)" %
                    (result.get("RMSE").mean(), result.get("RMSE").std()))

        logger.info("External validation: ")
        logger.info("\tR2 Scores: %0.3f" % result.get("Val_R2"))
        logger.info("\tRMSE Scores: %0.3f" % result.get("Val_RMSE"))
        return model
    else:
        r2_list = []
        rmse_list = []
        val_r2_list = []
        val_rmse_list = []
        # model_params = cfg.model_params.get(model_type).get(param_name)
        for i in range(train_times):
            result, model = train_model(X, y, model_type=model_type, params=model_params)
            logger.info(f"Epoch {i + 1}:")
            logger.info("\tR2 Scores: %0.4f (+/- %0.2f)" %
                        (result.get("R2").mean(), result.get("R2").std()))
            r2_list.append(result.get("R2").mean())
            logger.info("\tRMSE Scores: %0.4f (+/- %0.2f)" %
                        (result.get("RMSE").mean(), result.get("RMSE").std()))
            rmse_list.append(result.get("RMSE").mean())
            # print("\tNRMSE Scores: %0.4f (+/- %0.2f)" %
            #       (processed_data.get("NRMSE").mean(), processed_data.get("NRMSE").std()))

            logger.info("Validation: ")
            logger.info("\tR2 Scores: %0.4f" % result.get("Val_R2"))
            val_r2_list.append(result.get("Val_R2"))
            logger.info("\tRMSE Scores: %0.4f" % result.get("Val_RMSE"))
            val_rmse_list.append(result.get("Val_RMSE"))
            # print("\tNRMSE Scores: %0.4f" % processed_data.get("Val_NRMSE"))

        logger.info("\tSummery:")
        logger.info("\t\tTotal R2 Scores: %0.3f" % np.array(r2_list).mean())
        logger.info("\t\tTotal RMSE Scores: %0.3f" % np.array(rmse_list).mean())
        logger.info("\t\tTotal Validation R2 Scores: %0.3f" % np.array(val_r2_list).mean())
        logger.info("\t\tTotal Validation RMSE Scores: %0.3f" % np.array(val_rmse_list).mean())
        return None
