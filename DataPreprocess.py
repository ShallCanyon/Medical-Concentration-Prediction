import global_config as cfg
import pandas as pd
import numpy as np
from deepchem import feat
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def get_brainblood_csv(workbookpath, csvfilepath):
    # excel_df = pd.read_excel(workbookpath, index_col=[0, 1], engine='openpyxl')
    # # column_list = excel_df.columns.to_list()
    # # print(column_list)
    # blood_df = excel_df.loc[:, excel_df.columns.str.startswith('blood mean')]
    # brain_df = excel_df.loc[:, excel_df.columns.str.startswith('brain mean')]
    # # print(blood_df.columns.to_list())
    # # print(brain_df.columns.to_list())
    # df = pd.concat([blood_df, brain_df], axis=1)
    # df.to_csv(csvfilepath, encoding='utf-8')
    select_multiple_organs_data(workbookpath, csvfilepath, organs_name=['blood', 'brain'])


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


def select_multiple_organs_data(root_filepath: str, target_csvfilepath: str, organs_name: list):
    """
    筛选多个器官的所有浓度数据

    :param root_filepath: 源文件路径
    :param target_csvfilepath: 目标csv文件路径
    :param organs_name: 选定的器官名列表
    """
    if len(organs_name) == 0:
        raise ValueError("Empty list error")
    if root_filepath.endswith('xlsx') or root_filepath.endswith('xls'):
        raw_df = pd.read_excel(root_filepath, index_col=[0, 1], engine='openpyxl')
    elif root_filepath.endswith('csv'):
        raw_df = pd.read_csv(root_filepath, index_col=[0, 1])
    df_list = []
    for organ_name in organs_name:
        df = raw_df.loc[:, raw_df.columns.str.startswith(f'{organ_name} mean')]
        df_list.append(df)
    df = pd.concat(df_list, axis=1)
    df.to_csv(target_csvfilepath, encoding='utf-8')


def select_certain_time_organ_data(root_filepath: str, organ_name: str, certain_time: int):
    """
    筛选出指定器官在指定时间的浓度数据

    :param root_filepath: 源文件路径
    :param organ_name: 选定的器官名
    :param certain_time: 选定的时间点
    :return: 筛选得到的浓度数据dataframe
    """
    if root_filepath.endswith('xlsx') or root_filepath.endswith('xls'):
        raw_df = pd.read_excel(root_filepath, index_col=[0, 1], engine='openpyxl')
    elif root_filepath.endswith('csv'):
        raw_df = pd.read_csv(root_filepath, index_col=[0, 1])
    organ_df = raw_df.loc[:, raw_df.columns.str.startswith(f'{organ_name.lower()} mean{certain_time}min')]
    return organ_df


def select_max_organ_data(root_filepath: str, target_csvfilepath: str, organ_name: str):
    """
    根据选定的器官名，筛选出源csv文件中每个药物在该器官的最大浓度以及触及时间，并保存到目标csv文件中

    :param root_filepath: 源文件路径
    :param target_csvfilepath: 目标csv文件路径
    :param organ_name: 选定的器官名
    """
    # 读取数据并选择指定器官的全部浓度数据
    if root_filepath.endswith('xlsx') or root_filepath.endswith('xls'):
        raw_df = pd.read_excel(root_filepath, index_col=[0, 1], engine='openpyxl')
    elif root_filepath.endswith('csv'):
        raw_df = pd.read_csv(root_filepath, index_col=[0, 1])
    organ_df = raw_df.loc[:, raw_df.columns.str.startswith(f'{organ_name.lower()} mean')]
    # 保存每个药物到达最大浓度的数据
    max_concentration2time = dict()

    # 遍历每一款药物（index同时记录了文献编号和SMILES）
    for index, row_data in organ_df.iterrows():
        # 去除没有数据的列
        row_data = row_data.dropna()
        if row_data.empty:
            continue
        else:
            # 用于保存每个浓度数据与时间的对应关系
            num2time = dict()
            # 转换Series为Dataframe
            row_data = row_data.to_frame()
            row_data = pd.DataFrame(row_data.values.T, columns=row_data.index)
            for column in row_data.columns.to_list():
                concentration_num = float(row_data[column].values[0])
                # 将时间与浓度数据作为键值对保存到字典中
                num2time[column.split(" ")[1].replace('mean', '')] = concentration_num
        sorted_data = sorted(num2time.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        # 保存药物索引与最大浓度数据
        max_concentration2time[index] = sorted_data[0]
    # 将字典转换成Dataframe所需的列表格式
    max_data_list = []
    for key, value in max_concentration2time.items():
        index = key[0]
        smiles = key[1]
        time = value[0]
        concentration_num = value[1]
        max_data_list.append([index, smiles, concentration_num, time])
    df = pd.DataFrame(data=max_data_list,
                      columns=['Compound index', 'SMILES', 'Max Concentration', 'Reach time'])
    df.to_csv(target_csvfilepath, index=False)


def calculate_desc(datasrc, Mordred=True, MACCS=False, ECFP=False):
    """
    从datasrc中的SMILES计算描述符并返回带描述符的datasrc数据
    :param datasrc: 需要计算的带SMILES的数据源，类型为str（指向数据源的csv文件）或者Dataframe或Series
    :param Mordred: 启动Mordred描述符计算
    :param MACCS: 启动MACCS分子指纹计算
    :param ECFP: 启动ECFP分子指纹计算
    :return: 带描述符的datasrc数据
    """
    if isinstance(datasrc, str):
        df = pd.read_csv(datasrc)
    elif isinstance(datasrc, pd.DataFrame) or isinstance(datasrc, pd.Series):
        df = datasrc
    else:
        raise ValueError("错误的datasrc类型")

    if isinstance(datasrc, str) or isinstance(datasrc, pd.DataFrame):
        SMILES = df['SMILES']
    else:
        SMILES = df

    X = pd.DataFrame()
    # Mordred
    if Mordred:
        featurizer = feat.MordredDescriptors(ignore_3D=True)
        X1 = []
        for smiles in SMILES:
            X1.append(featurizer.featurize(smiles)[0])
        X1 = pd.DataFrame(data=X1)
        X = pd.concat([X, X1], axis=1)
    # MACCS
    if MACCS:
        X2 = []
        featurizer = feat.MACCSKeysFingerprint()
        for smiles in SMILES:
            X2.append(featurizer.featurize(smiles)[0])
        X2 = pd.DataFrame(data=X2)
        X = pd.concat([X, X2], axis=1)
    # ECFP
    if ECFP:
        X3 = []
        featurizer = feat.CircularFingerprint(size=2048, radius=4)
        for smiles in SMILES:
            X3.append(featurizer.featurize(smiles)[0])
        X3 = pd.DataFrame(data=X3)
        X = pd.concat([X, X3], axis=1)
    if not X.empty:
        # blood = df['Blood']
        # brain = df['Brain']
        # ratio = df['Brain/Blood']
        # df = pd.DataFrame(data=X)
        # df.insert(0, 'SMILES', SMILES)
        # df.insert(1, 'Blood', blood)
        # df.insert(2, 'Brain', brain)
        # df.insert(3, 'Ratio', ratio)

        # X.insert(0, 'SMILES', df['SMILES'])
        # X.insert(1, 'data', df.iloc[:, [False, False, True]])

        # df = pd.merge(df, X)
        df = pd.concat([df, X], axis=1)
        return df
        # df.to_csv(dstfile, index=False)
    else:
        raise ValueError("Empty dataframe")


def get_X_Y_by_ratio(csvfile):
    df = pd.read_csv(csvfile)
    print(df.shape)
    # 去除含有无效值的列
    # df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
    # df = df.dropna(axis=0, how='any')
    df = clean_desc_dataframe(df)
    print(df.shape)
    X = df.drop(['SMILES', 'Blood', 'Brain', 'Ratio'], axis=1)
    X = StandardScaler().fit_transform(X)
    # print(len(X))
    blood_y = df['Blood'].ravel()
    brain_y = df['Brain'].ravel()
    ratio_y = df['Ratio'].ravel()
    SMILES = df['SMILES']
    return pd.DataFrame(X).astype('float64'), blood_y, brain_y, ratio_y, SMILES


def get_SMILE_Y(df, smile_column='SMILES', y_column='Max Concentration'):
    if isinstance(df, str):
        df = pd.read_csv(df)
    print(df.shape)
    # 去除含有无效值的列
    # df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
    # df = df.dropna(axis=0, how='any')
    df = clean_desc_dataframe(df)
    print(df.shape)
    # X = df.drop(['SMILES', 'Blood', 'Brain', 'Ratio'], axis=1)
    # X = StandardScaler().fit_transform(X)
    # # print(len(X))
    y = df[y_column].ravel()
    SMILES = df[smile_column]
    return SMILES, y


def get_single_column_data(csv_file, smile_col=1, label_col=2, desc_start_col=3):
    """
    读取只有一列回归数据的csv文件，默认第一二列为药物index和SMILES，第三列为数据标签，其后为数据特征
    :param csv_file: 只有一列回归数据的csv文件
    :param smile_col: SMILES所在列号
    :param label_col: 数据标签所在列号
    :param desc_start_col: 数据特征起始列号
    :return: 完成清洗的数据特征X, 回归数据y，SMILES
    """
    df = pd.read_csv(csv_file)
    df = clean_desc_dataframe(df)
    X = df.iloc[:, desc_start_col:]
    y = df.iloc[:, label_col]
    smiles = df.iloc[:, smile_col]
    return X, y, smiles


def split_null_from_data(df):
    """
    将数据中的空数据与其他数据分开成两份dataframe

    :param df: 包含空数据的源dataframe
    :return: 含数据的dataframe以及含空数据的dataframe
    """
    data_df = df.dropna(axis=0)
    empty_df = df.drop(index=data_df.index)
    return data_df.reset_index(drop=True), empty_df.reset_index(drop=True)


def clean_desc_dataframe(df: pd.DataFrame, axis=1, drop_duplicates=True) -> pd.DataFrame:
    """
    清除描述符dataframe中的无效数据，避免发生报错

    :param df: 包含无效数据的Dataframe
    :param axis: axis为1时清除掉包含无效数据的列（默认），0时清除行
    :param drop_duplicates: 是否丢弃重复的行
    :return: 完成清除的Dataframe
    """
    df = df.replace(["#NAME?", np.inf, -np.inf], np.nan)
    df = df.dropna(axis=axis)
    if drop_duplicates:
        df = df.drop_duplicates()
    return df
