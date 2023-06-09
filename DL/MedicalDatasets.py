import os.path

from learn2learn.data import MetaDataset
from torch.utils.data import TensorDataset
import DataPreprocess
from FeatureExtraction import FeatureExtraction
from DataPreprocess import get_single_column_data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch


class MedicalDatasets:
    def __init__(self, csv_filepath, folder_path):
        """

        :param csv_filepath: 指定时间段的药物浓度数据
        :param folder_path: 用于存储各项数据的目录
        """
        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.csv_filepath = csv_filepath
        self.folder_path = folder_path
        self.desc_num = 0

    def get_single_organ_tensor(self, test_size=0.1):
        x, y, _ = get_single_column_data(self.csv_filepath)
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

    def transform_organs_data(self, double_index=True, overwrite=False):
        """
        读取保存全部器官浓度数据的csv文件，并逐个处理为单一器官的dataframe并以{器官名：器官df}的格式保存到npy文件中
        :param double_index:
        :param overwrite: 是否覆盖已有的npy文件
        :return: 保存所有器官及其df的字典
        """
        # 路径初始化
        npy_file = f'{self.folder_path}\\multi_organ.npy'
        desc_file = f'{self.folder_path}\\Mordred_md.csv'
        mordred_50_tuned_index = f'{self.folder_path}\\mordred_50_tuned_index.npy'
        mordred_100_tuned_index = f'{self.folder_path}\\mordred_100_tuned_index.npy'
        if overwrite or not os.path.exists(npy_file):
            # 读取浓度数据，并获取分子描述符
            df = pd.read_csv(self.csv_filepath)
            # df = clean_desc_dataframe(df)
            smiles = pd.DataFrame({'SMILES': df.iloc[:, 1]})
            # 计算SMILES的Mordred描述符，然后保存到desc_file文件中方便再次读取
            if not os.path.exists(desc_file):
                Modred_Desc = DataPreprocess.calculate_desc(smiles).iloc[:, 1:]
                Modred_Desc.to_csv(desc_file, index=False)
            else:
                Modred_Desc = pd.read_csv(desc_file)

            # 预处理数据集的x和y
            sc = StandardScaler()
            Modred_Desc = pd.DataFrame(sc.fit_transform(Modred_Desc), columns=Modred_Desc.columns)
            Modred_Desc = DataPreprocess.clean_desc_dataframe(Modred_Desc, drop_duplicates=False)
            organs_y = df.iloc[:, 2:]

            # 保存所有器官的描述符以及浓度数据的字典
            datasets = {}
            # 特征提取的列索引，从文件中读取，若不存在则进行特征提取后写入文件中
            desc_50_idx_list = []
            desc_100_idx_list = []
            if os.path.exists(mordred_50_tuned_index):
                desc_50_idx_list = np.load(mordred_50_tuned_index).tolist()
                print("Length of 50 desc list: ", len(desc_50_idx_list))
            if os.path.exists(mordred_100_tuned_index):
                desc_100_idx_list = np.load(mordred_100_tuned_index).tolist()
                print("Length of 100 desc list: ", len(desc_100_idx_list))

            for index, col in organs_y.iteritems():
                organ_name = index.split()[0]
                concentration_data = pd.DataFrame({'Concentration': col})
                # concentration_data = pd.Series({'Concentration': col})
                # 保存50个筛选特征索引
                if len(desc_50_idx_list) == 0:
                    desc_50_idx_list = FeatureExtraction(Modred_Desc, concentration_data.fillna(value=0)). \
                        feature_extraction(TBE=True, returnIndex=True)
                    print("Length of 50 desc list: ", len(desc_50_idx_list))
                    np.save(mordred_50_tuned_index, desc_50_idx_list)
                # 保存100个筛选特征索引
                if len(desc_100_idx_list) == 0:
                    desc_100_idx_list = FeatureExtraction(Modred_Desc,
                                                          concentration_data.fillna(value=0),
                                                          RFE_features_to_select=100) \
                        .feature_extraction(TBE=True, returnIndex=True)
                    print("Length of 100 desc list: ", len(desc_100_idx_list))
                    np.save(mordred_100_tuned_index, desc_100_idx_list)

                if double_index:
                    Modred_Desc = Modred_Desc.loc[:, desc_100_idx_list]
                else:
                    Modred_Desc = Modred_Desc.loc[:, desc_50_idx_list]
                organ_df = pd.concat([smiles, concentration_data, Modred_Desc], axis=1)
                # 根据浓度数据列的空数据抛弃行数据
                organ_df = organ_df.dropna(subset=['Concentration'])
                organ_df.reset_index(inplace=True, drop=True)
                datasets[organ_name] = organ_df
            np.save(npy_file, datasets)
        else:
            datasets = np.load(npy_file, allow_pickle=True).item()
        return datasets

        # # Wrap the datasets using MetaDataset
        # meta_train_dataset = MetaDataset(train_dataset)
        # meta_test_dataset = MetaDataset(test_dataset)

    def save_dataframes2Tensor(self, df_map: dict):
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
            torch.save(dataset, f'{self.folder_path}\\{name}_{count}_dataset.pt')
