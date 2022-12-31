import pandas as pd
import openpyxl
import os
import deepchem as dc
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error, median_absolute_error, mean_squared_log_error, mean_absolute_error, r2_score


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
                    ratio2time[column.split(" ")[1].replace('mean', '')] = brain_num / blood_num
        # compound_ratio[index] = ratio2time
        # 降序排序并获取第一个最大值
        compound_ratio[index] = sorted(ratio2time.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[0]
    # 将字典转换成Dataframe所需的列表格式
    max_ratio_list = []
    for key, value in compound_ratio.items():
        index = key[0]
        smiles = key[1]
        time = value[0]
        ratio = value[1]
        max_ratio_list.append([index, smiles, ratio, time])
    df = pd.DataFrame(data=max_ratio_list, columns=['Compound index', 'SMILES', 'Max(Brain/Blood)', 'Reach time'])
    # print(df)
    df.to_csv(ratio_csvfilepath, index=False)


def RegressionPredicting(csvfile):
    df = pd.read_csv(csvfile)
    featurizer = dc.feat.MordredDescriptors(ignore_3D=True)
    SMILES = df['SMILES']
    X = []
    for smiles in SMILES:
        X.append(featurizer.featurize(smiles)[0])
    X = MinMaxScaler().fit_transform(X)
    print(len(X))
    y = df['Max(Brain/Blood)'].ravel()

    print("Start training model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    model = XGBRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    MSE = mean_squared_error(y_test, y_pred)
    # MSLE = mean_squared_log_error(y_test, y_pred)
    MAE = mean_absolute_error(y_test, y_pred)
    Median = median_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("MSE: ", MSE)
    # print("MSLE: ", MSLE)
    print("MAE: ", MAE)
    print("Median: ", Median)
    print("r2: ", r2)

    # features = featurizer.featurize(SMILES[0])
    # print(features[0].shape)
    # pd.DataFrame(data=features, columns=dc.feat.Mor)

if __name__ == '__main__':
    filetime = "20221221"
    workbookpath = f"./result/{filetime}/数据表汇总.xlsx"
    raw_csvfilepath = f"./result/{filetime}/BrainBlood.csv"
    ratio_csvfilepath = f"./result/{filetime}/MaxBrainBloodRatio.csv"
    generate_raw_data = False

    print("Running...")
    if not os.path.exists(raw_csvfilepath) or generate_raw_data:
        print("Getting blood brain file...")
        get_brainblood_csv(workbookpath, raw_csvfilepath)
    print("Calculating blood brain ratio...")
    # calculate_blood_brain_ratio(raw_csvfilepath, ratio_csvfilepath)
    RegressionPredicting(ratio_csvfilepath)
