import pandas as pd
import numpy as np
import openpyxl
import os
import deepchem as dc
from sklearn.preprocessing import MinMaxScaler
# from lightgbm import LGBMRegressor
import lightgbm as lgb
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


def get_brainblood_csv(workbookpath, csvfilepath):
    excel_df = pd.read_excel(workbookpath, index_col=[0, 1], engine='openpyxl')
    # column_list = excel_df.columns.to_list()
    # print(column_list)
    blood_df = excel_df.loc[:, excel_df.columns.str.startswith('blood mean')]
    brain_df = excel_df.loc[:, excel_df.columns.str.startswith('brain mean')]
    print(blood_df.columns.to_list())
    print(brain_df.columns.to_list())
    df = pd.concat([blood_df, brain_df], axis=1)
    df.to_csv(csvfilepath, encoding='utf-8')


def calculate_blood_brain_ratio(raw_csvfilepath, ratio_csvfilepath):
    raw_df = pd.read_csv(raw_csvfilepath, index_col=[0, 1])
    blood_df = raw_df.loc[:, raw_df.columns.str.startswith('blood mean')]
    brain_df = raw_df.loc[:, raw_df.columns.str.startswith('brain mean')]
    # 以{(化合物文献号，SMILE) -> {浓度数据}}的格式存储数据
    compound_ratio = dict()
    for index, blood_row_data in blood_df.iterrows():
        # 血液行数据
        blood_row_data = blood_row_data.dropna()
        # 脑部行数据
        brain_row_data = brain_df.loc[index[0]].dropna(axis=1, how='all')
        # 任意一个器官内数据为空，跳过
        if brain_row_data.empty or blood_row_data.empty:
            continue
        else:
            # 以{(时间) -> (脑血浓度比)}的格式存储数据
            ratio2time = dict()
            # 转换series为dataframe
            blood_row_data = blood_row_data.to_frame()
            blood_row_data = pd.DataFrame(blood_row_data.values.T, columns=blood_row_data.index)

            for column in blood_row_data.columns.to_list():
                # 获取血液浓度
                blood_num = float(blood_row_data[column].values[0])
                # 拆分列头以获取时间点，组合成脑部浓度数据时间点
                tgt_col = 'brain ' + column.split(" ")[1]
                # 判断该脑部数据时间点是否存在
                if tgt_col in brain_row_data.columns.to_list():
                    # 获取脑部浓度
                    brain_num = float(brain_row_data[tgt_col].values[0])
                    brainbloodratio = brain_num / blood_num
                    # 按照脑部浓度、血液浓度和脑血浓度比3种数据以列表格式保存到字典中
                    ratio2time[column.split(" ")[1].replace('mean', '')] = [brain_num, blood_num, brainbloodratio]
        # kv[1][2]指定为以脑血浓度比进行降序排序
        sorted_data = sorted(ratio2time.items(), key=lambda kv: (kv[1][2], kv[0]), reverse=True)
        # 获取最大脑血浓度比的数据
        compound_ratio[index] = sorted_data[0]
    # 将字典转换成Dataframe所需的列表格式
    max_ratio_list = []
    for key, value in compound_ratio.items():
        index = key[0]
        smiles = key[1]
        time = value[0]
        brain_num = value[1][0]
        blood_num = value[1][1]
        ratio = value[1][2]
        max_ratio_list.append([index, smiles, brain_num, blood_num, ratio, time])
    df = pd.DataFrame(data=max_ratio_list,
                      columns=['Compound index', 'SMILES', 'Brain', 'Blood', 'Brain/Blood', 'Reach time'])
    df.to_csv(ratio_csvfilepath, index=False)


def calculate_desc(srcfile, dstfile):
    df = pd.read_csv(srcfile)
    # Mordred
    featurizer = dc.feat.MordredDescriptors(ignore_3D=True)
    SMILES = df['SMILES']
    X1 = []
    for smiles in SMILES:
        X1.append(featurizer.featurize(smiles)[0])
    X1 = pd.DataFrame(data=X1)
    # MACCS
    X2 = []
    featurizer = dc.feat.MACCSKeysFingerprint()
    for smiles in SMILES:
        X2.append(featurizer.featurize(smiles)[0])
    X2 = pd.DataFrame(data=X2)
    # ECFP
    X3 = []
    featurizer = dc.feat.CircularFingerprint(size=2048, radius=4)
    for smiles in SMILES:
        X3.append(featurizer.featurize(smiles)[0])
    X3 = pd.DataFrame(data=X3)
    X = pd.concat([X1, X2, X3], axis=1)

    blood = df['Blood']
    brain = df['Brain']
    ratio = df['Brain/Blood']
    df = pd.DataFrame(data=X)
    df.insert(0, 'SMILES', SMILES)
    df.insert(1, 'Blood', blood)
    df.insert(2, 'Brain', brain)
    df.insert(3, 'Ratio', ratio)
    df.to_csv(dstfile, index=False)


def get_X_Y(csvfile):
    df = pd.read_csv(csvfile)
    # 去除含有无效值的列
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
    df = df.dropna(axis=0, how='any')

    X = df.drop(['SMILES', 'Blood', 'Brain', 'Ratio'], axis=1)
    X = MinMaxScaler().fit_transform(X)
    # print(len(X))
    blood_y = df['Blood'].ravel()
    brain_y = df['Brain'].ravel()
    ratio_y = df['Ratio'].ravel()
    SMILES = df['SMILES']
    return pd.DataFrame(X).astype('float64'), blood_y, brain_y, ratio_y, SMILES


def train_model(X, y, model_type=None, param_name=None, cv_times=10):
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

    preds = model.predict(X_val)
    val_r2 = r2_score(y_val, preds)
    val_rmse = np.sqrt(mean_squared_error(y_val, preds))
    # print("Validation r2: ", val_r2)
    # print("Validation rmse: ", val_rmse)

    return r2_scores, rmse_scores, val_r2, val_rmse


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
