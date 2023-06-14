import os.path

from torch.utils.data import TensorDataset
from DataPreprocess.FeatureExtraction import FeatureExtraction
from DataPreprocess import DataPreprocess
from DataPreprocess.PadelpyCall import PadelpyCall
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch


class MedicalDatasets:
    def __init__(self, organ_data_filepath, folder_path):
        """
        :param organ_data_filepath: 指定时间段的药物浓度数据
        :param folder_path: 用于存储各项数据的目录
        """
        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.organ_data_filepath = organ_data_filepath
        self.folder_path = folder_path
        self.desc_num = 0

    def transform_organs_data(self, desc_file='Mordred_md.csv', FP=False, double_index=True, overwrite=False):
        """
        读取保存全部器官浓度数据的csv文件，并逐个处理为单一器官的dataframe并以{器官名：器官df}的格式保存到npy文件中
        :param desc_file:
        :param FP: 启动分子指纹处理
        :param double_index: 是否读取双倍（100个）的特征索引
        :param overwrite: 是否覆盖已有的npy文件
        :return: 保存所有器官及其df的字典
        """
        # 路径初始化
        npy_file = os.path.join(self.folder_path, 'multi_organ.npy')
        desc_file = os.path.join(self.folder_path, desc_file)
        # mordred_50_tuned_index = os.path.join(self.folder_path, 'mordred_50_tuned_index.npy')
        # mordred_100_tuned_index = os.path.join(self.folder_path, 'mordred_100_tuned_index.npy')

        if overwrite or not os.path.exists(npy_file):
            # 读取浓度数据，并获取分子描述符
            df = pd.read_csv(self.organ_data_filepath)
            # df = clean_desc_dataframe(df)
            smiles = pd.DataFrame({'SMILES': df.iloc[:, 1]})
            if not os.path.exists(desc_file):
                # 计算SMILES的描述符，然后保存到mol_Desc文件中方便再次读取
                if FP:  # 分子指纹
                    pc = PadelpyCall(smi_dir="/Data/DL/Datasets/479smiles.smi",
                                     fp_xml_dir="./fingerprints_xml/*.xml")
                    mol_Desc = pc.CalculateFP(['EState', 'MACCS', 'KlekotaRoth', 'PubChem'])
                else:  # 分子描述符
                    mol_Desc = DataPreprocess.calculate_desc(smiles)
                    mol_Desc.to_csv(desc_file, index=False)
            else:
                mol_Desc = pd.read_csv(desc_file)
            # 读取纯特征部分为x
            mol_Desc = mol_Desc.iloc[:, 1:]
            # 预处理数据集的x和y
            sc = StandardScaler()
            mol_Desc = pd.DataFrame(sc.fit_transform(mol_Desc), columns=mol_Desc.columns)
            mol_Desc = DataPreprocess.clean_desc_dataframe(mol_Desc, drop_duplicates=False)
            organs_y = df.iloc[:, 2:]

            # 保存所有器官的描述符以及浓度数据的总字典
            datasets = {}
            # 特征提取的列索引，从文件中读取，若不存在则进行特征提取后写入文件中
            # desc_50_idx_list = []
            # desc_100_idx_list = []
            # if os.path.exists(mordred_50_tuned_index):
            #     desc_50_idx_list = np.load(mordred_50_tuned_index).tolist()
            #     print("Length of 50 desc list: ", len(desc_50_idx_list))
            # if os.path.exists(mordred_100_tuned_index):
            #     desc_100_idx_list = np.load(mordred_100_tuned_index).tolist()
            #     print("Length of 100 desc list: ", len(desc_100_idx_list))

            # 处理每一种器官的浓度数据
            for index, col in organs_y.iteritems():
                organ_name = index.split()[0]
                concentration_data = pd.DataFrame({'Concentration': col})
                # concentration_data = pd.Series({'Concentration': col})
                """
                    若特征索引不存在，则进行特征筛选，分别获得50个和100个特征的索引
                """
                # 保存50个筛选特征索引
                # if len(desc_50_idx_list) == 0:
                if not double_index:
                    desc_50_idx_list = FeatureExtraction(mol_Desc,
                                                         concentration_data.fillna(value=0),
                                                         RFE_features_to_select=50). \
                        feature_extraction(TBE=True, returnIndex=True)
                    # print("Length of 50 desc list: ", len(desc_50_idx_list))
                    # np.save(mordred_50_tuned_index, desc_50_idx_list)
                    x = mol_Desc.loc[:, desc_50_idx_list]
                # 保存100个筛选特征索引
                else:
                    desc_100_idx_list = FeatureExtraction(mol_Desc,
                                                          concentration_data.fillna(value=0),
                                                          RFE_features_to_select=100) \
                        .feature_extraction(TBE=True, returnIndex=True)
                    # print("Length of 100 desc list: ", len(desc_100_idx_list))
                    # np.save(mordred_100_tuned_index, desc_100_idx_list)
                    x = mol_Desc.loc[:, desc_100_idx_list]
                # if double_index:
                #     x = mol_Desc.loc[:, desc_100_idx_list]
                # else:
                #     x = mol_Desc.loc[:, desc_50_idx_list]

                # 合并SMILES、浓度数据和筛选完成的特征
                organ_df = pd.concat([smiles, concentration_data, x], axis=1)
                # 根据浓度数据列的空数据抛弃行数据
                organ_df = organ_df.dropna(subset=['Concentration'])
                organ_df.reset_index(inplace=True, drop=True)
                # 按照器官名添加到总字典中
                datasets[organ_name] = organ_df
            # 保存字典
            np.save(npy_file, datasets)
        # 总字典存在，直接读取
        else:
            datasets = np.load(npy_file, allow_pickle=True).item()
        return datasets

    def save_df2TensorDataset(self, df_map: dict):
        """
        输入字典数据，将字典内的dataframe转换成TensorDataset并保存
        :param df_map: 输入的字典数据
        """
        for name, df in df_map.items():
            # df = DataPreprocess.clean_desc_dataframe(df)
            x = df.iloc[:, 2:]
            y = df['Concentration']
            if x.shape[0] != y.shape[0]:
                raise ValueError("x and y having different counts")
            count = y.shape[0]
            x = torch.tensor(x.values).to(self.__device)
            y = torch.tensor(y.values).resize_(count, 1).to(self.__device)
            dataset = TensorDataset(x, y)
            torch.save(dataset, os.path.join(self.folder_path, f'{name}_{count}_dataset.pt'))

    def get_single_organ_tensor(self, test_size=0.1):
        x, y, _ = DataPreprocess.get_single_column_data(self.organ_data_filepath)
        sc = StandardScaler()
        x = pd.DataFrame(sc.fit_transform(x))

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
        sample_num, self.desc_num = x.shape[0], x.shape[1]

        # Prepare your data as PyTorch tensors
        x_train, y_train = torch.Tensor(x_train.values).to(self.__device), \
            torch.Tensor(y_train.values).resize_(y_train.shape[0], 1).to(self.__device)
        x_test, y_test = torch.Tensor(x_test.values).to(self.__device), \
            torch.Tensor(y_test.values).resize_(y_test.shape[0], 1).to(self.__device)

        # Create PyTorch datasets
        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)

        return train_dataset, test_dataset
