import global_config as cfg
import optuna
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from xgboost.sklearn import XGBRegressor
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.neural_network import MLPRegressor as MLP
# from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from data_preprocess import get_X_Y_by_ratio


class ParamsTuning:
    """
        用于机器学习模型调参的类，需要调整的参数已经预设在类中
    """
    def __init__(self, model_type, n_trials=20, cv=10, study_name=None):
        """
        用于机器学习模型调参的类，需要调整的参数已经预设在类中

        初始化需要训练的模型、调参的次数、交叉验证的次数以及调参任务名
        :param model_type: 需要训练的模型，需要调用config中的模型枚举
        :param n_trials: 调参的次数
        :param cv: 交叉验证的次数
        :param study_name: 调参任务名
        """
        self.directions = ["minimize", "maximize"]
        self.direction = self.directions[1]
        self.model_type = model_type
        self.n_trials = n_trials
        self.study_name = study_name
        self.cv = cv
        optuna.logging.enable_propagation()  # Propagate logs to the root logger.
        optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

    def set_direction(self, direction):
        """
        设置调参的优化方向（默认使用r2与最大化方向）
        :param direction: 优化方向，类型为字符串
        :return: None
        """
        if type(direction) is not str and direction not in self.directions:
            raise ValueError("Parameter 'direction' should be 'minimize' or 'maximize'")
        self.direction = direction

    def tune_params(self, X, y):
        """
        根据输入的数据进行模型调参，返回study对象，通过访问study.best_params获取最佳参数
        :param X: 数据特征
        :param y: 数据标签
        :return: study对象
        """
        study = optuna.create_study(direction=self.direction, study_name=self.study_name)
        func = lambda trial: self.objective(trial, X, y)
        study.optimize(func, n_trials=self.n_trials)
        return study

    def objective(self, trial, X, y):
        """
        调参的核心方法，使用n折交叉验证获取一组参数下的模型平均结果并返回
        :param trial: optuna的trial对象
        :param X: 数据特征
        :param y: 数据标签
        :return: 模型训练结果
        """
        param_grid = self.__getParamsGrid__(trial)
        cv = KFold(n_splits=self.cv, shuffle=True)
        cv_scores = np.empty(self.cv)

        for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if self.model_type == cfg.model_enum[0]:
                model = XGBRegressor(**param_grid)
                model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=False)
            elif self.model_type == cfg.model_enum[1]:
                model = lgb.sklearn.LGBMRegressor(**param_grid)
                # callbacks = [lgb.early_stopping(100, verbose=0), lgb.log_evaluation(period=0)]
                callbacks = [lgb.log_evaluation(period=0)]
                model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=callbacks)
            elif self.model_type == cfg.model_enum[2]:  # SVM
                model = SVR(**param_grid)
                model.fit(X_train, y_train)
            elif self.model_type == cfg.model_enum[3]:  # RF
                model = RF(**param_grid)
                model.fit(X_train, y_train)
            elif self.model_type == cfg.model_enum[4]:  # MLP
                model = MLP(**param_grid)
                model.fit(X_train, y_train)
            preds = model.predict(X_test)
            cv_scores[idx] = r2_score(y_test, preds)
            # cv_scores[idx] = np.sqrt(mean_squared_error(y_test, preds)) / (y_test.max() - y_test.min())
            # cv_scores[idx] = np.sqrt(mean_squared_error(y_test, preds))
        return np.mean(cv_scores)

    def __getParamsGrid__(self, trial):
        """
        获取模型调参对应的参数矩阵
        :param trial: optuna调参的trial对象
        :return: 参数矩阵
        """
        if self.model_type == cfg.model_enum[0]:
            param_grid = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 2000, step=50),
                "learning_rate": trial.suggest_float('learning_rate', 0.005, 0.03, step=0.001),
                "max_depth": trial.suggest_int("max_depth", 0, 30),
                'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
                'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
                'gamma': trial.suggest_int("gamma", 0, 20, step=1),
                'colsample_bytree': trial.suggest_float("colsample_bytree", 0, 1, step=0.1),
                'colsample_bylevel': trial.suggest_float("colsample_bylevel", 0, 1, step=0.1),
                'colsample_bynode': trial.suggest_float("colsample_bynode", 0, 1, step=0.1),
                "n_jobs": trial.suggest_categorical("n_jobs", [-1])
            }
        # LGBM params
        elif self.model_type == cfg.model_enum[1]:
            param_grid = {
                "boosting_type": trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
                "max_depth": trial.suggest_int("max_depth", 1, 30),
                # "learning_rate": trial.suggest_categorical('learning_rate',
                #                                            [0.005, 0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02, 0.023,
                #                                             0.025, 0.028, 0.03]),
                "learning_rate": trial.suggest_float('learning_rate', 0.005, 0.03, step=0.001),
                "n_estimators": trial.suggest_int("n_estimators", 50, 3000, step=50),
                "objective": trial.suggest_categorical('objective', ['regression']),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
                # 'colsample_bytree': trial.suggest_float("colsample_bytree", 0, 1, step=0.1),
                'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
                'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
                # 'feature_fraction': trial.suggest_categorical('feature_fraction', [0.5])
                # 'verbose': trial.suggest_categorical('verbose', [-1])
            }
        elif self.model_type == cfg.model_enum[2]:  # SVM
            param_grid = {
                "C": trial.suggest_float('C', 0.1, 10),
                'gamma': trial.suggest_categorical("gamma", ['scale', 'auto']),
                'tol': trial.suggest_categorical("tol", [1e-2, 1e-3, 1e-4]),
                'max_iter': trial.suggest_categorical("max_iter", [1000, 5000, 10000]),
                'epsilon': trial.suggest_float("epsilon", 0.1, 1.0)
            }
        elif self.model_type == cfg.model_enum[3]:  # RF
            param_grid = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 3000, step=100),
                "max_depth": trial.suggest_int("max_depth", 1, 30),
                "n_jobs": trial.suggest_categorical("n_jobs", [-1])
            }
        elif self.model_type == cfg.model_enum[4]:  # MLP
            param_grid = {
                "hidden_layer_sizes": trial.suggest_categorical("hidden_layer_sizes", [(50,), (100,), (150,), (200,)]),
                "activation": trial.suggest_categorical("activation", ['tanh', 'relu']),
                "solver": trial.suggest_categorical("solver", ['lbfgs', 'sgd', 'adam']),
                "learning_rate": trial.suggest_categorical("learning_rate", ['constant', 'invscaling', 'adaptive']),
                "learning_rate_init": trial.suggest_categorical("learning_rate_init", [0.001, 0.002, 0.005, 0.01, 0.02]),
                "early_stopping": True,
                "max_iter": trial.suggest_categorical("max_iter", [500, 1000, 2000, 5000]),
            }
        return param_grid
