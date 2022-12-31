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
from sklearn.metrics import mean_squared_error, median_absolute_error, mean_squared_log_error, mean_absolute_error, r2_score
import sklearn.metrics
import global_config as cfg

def main():
    # 原始的集成数据集
    workbookpath = cfg.workbookpath
    # 从原始数据集中挑选出脑部与血液浓度的数据集
    raw_csvfilepath = cfg.raw_csvfilepath
    # 计算得到最大脑血比的数据集
    ratio_csvfilepath = cfg.ratio_csvfilepath
    # 计算出药物的Mordred描述符以及最大脑血比的数据集
    desc_csvfilepath = cfg.desc_csvfilepath
    generate_new_data = [False, False, False]
    regressor_type = 'LGBM'

    print("Running...")
    if not os.path.exists(raw_csvfilepath) or generate_new_data[0]:
        print("Getting blood brain file...")
        get_brainblood_csv(workbookpath, raw_csvfilepath)

    if not os.path.exists(ratio_csvfilepath) or generate_new_data[1]:
        print("Calculating blood brain ratio...")
        calculate_blood_brain_ratio(raw_csvfilepath, ratio_csvfilepath)

    if not os.path.exists(desc_csvfilepath) or generate_new_data[2]:
        print("Calculating descriptors...")
        calculate_desc(ratio_csvfilepath, desc_csvfilepath)

    X, blood_y, brain_y, ratio_y, SMILES = get_X_Y(desc_csvfilepath)
    feature_select = True
    if feature_select:
        # 特征筛选
        blood_X = X.iloc[:, cfg.blood_fea]
        brain_X = X.iloc[:, cfg.brain_fea]
    else:
        blood_X = X
        brain_X = X

    print("Start training model...")
    blood_r2_scores, blood_rmse_scores, brain_r2_scores,brain_rmse_scores = train_model(blood_X, brain_X, blood_y, brain_y, ratio_y, cfg.model_type)

    print("Blood data:")
    print("R2 Scores: %0.4f (+/- %0.2f)" %
        (blood_r2_scores.mean(), blood_r2_scores.std()))
    print("RMSE Scores: %0.4f (+/- %0.2f)" %
        (blood_rmse_scores.mean(), blood_rmse_scores.std()))

    print("Brain data:")
    print("R2 Scores: %0.4f (+/- %0.2f)" %
        (brain_r2_scores.mean(), brain_r2_scores.std()))
    print("RMSE Scores: %0.4f (+/- %0.2f)" %
        (brain_rmse_scores.mean(), brain_rmse_scores.std()))


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
        # print(sorted_data)
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
    df = pd.DataFrame(data=max_ratio_list, columns=['Compound index', 'SMILES', 'Brain', 'Blood', 'Brain/Blood', 'Reach time'])

    #     # 降序排序并获取第一个最大值
    #     compound_ratio[index] = sorted(ratio2time.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[0]
    # # 将字典转换成Dataframe所需的列表格式
    # max_ratio_list = []
    # for key, value in compound_ratio.items():
    #     index = key[0]
    #     smiles = key[1]
    #     time = value[0]
    #     ratio = value[1]
    #     max_ratio_list.append([index, smiles, ratio, time])
    # df = pd.DataFrame(data=max_ratio_list, columns=['Compound index', 'SMILES', 'Max(Brain/Blood)', 'Reach time'])
    # print(df)
    df.to_csv(ratio_csvfilepath, index=False)

def calculate_desc(srcfile, dstfile):
    df = pd.read_csv(srcfile)
    featurizer = dc.feat.MordredDescriptors(ignore_3D=True)
    SMILES = df['SMILES']
    X = []
    for smiles in SMILES:
        X.append(featurizer.featurize(smiles)[0])
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
    X = df.drop(['SMILES', 'Blood', 'Brain', 'Ratio'], axis=1)
    X = MinMaxScaler().fit_transform(X)
    # print(len(X))
    blood_y = df['Blood'].ravel()
    brain_y = df['Brain'].ravel()
    ratio_y = df['Ratio'].ravel()
    SMILES = df['SMILES']
    return pd.DataFrame(X), blood_y, brain_y, ratio_y, SMILES

def train_model(blood_X, brain_X, blood_y, brain_y, ratio_y, model_type=None, cv_times=5):
    if model_type is None:
        raise ValueError
    if model_type == cfg.model_enum[0]: #XGB
        params = cfg.model_params.get('XGB')
        blood_model = XGBRegressor(**params.get('blood_params'))
        brain_model = XGBRegressor(**params.get('brain_params'))
    elif model_type == cfg.model_enum[1]: #LGBM
        params = cfg.model_params.get('LGBM')
        blood_model = LGBMRegressor(**params.get('blood_params'))
        brain_model = LGBMRegressor(**params.get('brain_params'))
        
    cv = KFold(n_splits=cv_times, shuffle=True)

    blood_r2_scores = np.empty(cv_times)
    blood_rmse_scores = np.empty(cv_times)
    brain_r2_scores = np.empty(cv_times)
    brain_rmse_scores = np.empty(cv_times)
    
    # Blood
    for idx, (train_idx, test_idx) in enumerate(cv.split(blood_X, blood_y)):
        X_train, X_test = blood_X.iloc[train_idx], blood_X.iloc[test_idx]
        y_train, y_test = blood_y[train_idx], blood_y[test_idx]
        
        if model_type == cfg.model_enum[0]: #XGB
            blood_model.fit(X_train, y_train, eval_set=[
            (X_test, y_test)], early_stopping_rounds=100, verbose=False)
        elif model_type == cfg.model_enum[1]:   #LGBM
            callbacks = [lgb.log_evaluation(period=0)]
            blood_model.fit(X_train, y_train, eval_set=[
            (X_test, y_test)], callbacks=callbacks)
        
        preds = blood_model.predict(X_test)

        r2 = r2_score(y_test, preds)
        blood_r2_scores[idx] = r2

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        blood_rmse_scores[idx] = rmse
    # Brain
    for idx, (train_idx, test_idx) in enumerate(cv.split(brain_X, brain_y)):
        X_train, X_test = brain_X.iloc[train_idx], brain_X.iloc[test_idx]
        y_train, y_test = brain_y[train_idx], brain_y[test_idx]

        if model_type == cfg.model_enum[0]: #XGB
            brain_model.fit(X_train, y_train, eval_set=[
            (X_test, y_test)], early_stopping_rounds=100, verbose=False)
        elif model_type == cfg.model_enum[1]:   #LGBM
            callbacks = [lgb.log_evaluation(period=0)]
            brain_model.fit(X_train, y_train, eval_set=[
            (X_test, y_test)], callbacks=callbacks) 
        
        preds = brain_model.predict(X_test)

        r2 = r2_score(y_test, preds)
        brain_r2_scores[idx] = r2

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        brain_rmse_scores[idx] = rmse
    
    return blood_r2_scores, blood_rmse_scores, brain_r2_scores,brain_rmse_scores

if __name__ == '__main__':
    main()