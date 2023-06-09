import os.path
import subprocess
import glob
import global_config as cfg
import pandas as pd
from tqdm import tqdm
from DataLogger import DataLogger
from padelpy import padeldescriptor

logger = DataLogger(cfg.logger_filepath, 'Padel').getlog()


class PadelpyCall:
    def __init__(self, smi_dir, save_dir='./FP_result', maxruntime=30):
        """
        调用Padelpy进行分子指纹计算

        :param smi_dir: 记录待计算分子的SMILES的smi文件
        :param save_dir: 保存计算结果的目录
        :param maxruntime:
        """
        self.smi_dir = smi_dir
        self.save_dir = save_dir
        self.maxruntime = maxruntime

        self.index_name = 'SMILES'
        self.xml_files = glob.glob("./fingerprints_xml/*.xml")
        self.xml_files.sort()
        self.FP_list = ['AtomPairs2DCount',
                        'AtomPairs2D',
                        'EState',
                        'CDKextended',
                        'CDK',
                        'CDKgraphonly',
                        'KlekotaRothCount',
                        'KlekotaRoth',
                        'MACCS',
                        'PubChem',
                        'SubstructureCount',
                        'Substructure']
        self.fp = dict(zip(self.FP_list, self.xml_files))

    def CalculateFP(self, fingerprints):
        """
        遍历计算分子指纹，分别输出对应分子指纹的文件
        :param fingerprints:
        :return:
        """
        # 获取smi文件的SMILES，等待后续使用
        with open(self.smi_dir, "r") as file:
            lines = file.readlines()
        # 去除每行末尾的换行符，并创建一个包含每行内容的SMILES列表
        SMILES = [line.rstrip("\n") for line in lines]

        logger.info(f"Start calculating fingerprints of {self.smi_dir}")
        for fingerprint in tqdm(fingerprints, desc="Calculation process: "):
            if fingerprint not in self.FP_list:
                logger.error(f"Fingerprint error: {fingerprint} is not in fingerprint list")
                continue
            # fingerprint = 'Substructure'
            fingerprint_output_file = os.path.join(self.save_dir, ''.join([fingerprint, '.csv']))
            fingerprint_descriptor_types = self.fp[fingerprint]  # 解析文件地址

            padeldescriptor(mol_dir=self.smi_dir,
                            d_file=fingerprint_output_file,  # ex: 'Substructure.csv'
                            descriptortypes=fingerprint_descriptor_types,
                            detectaromaticity=True,
                            standardizenitro=True,
                            standardizetautomers=True,
                            threads=2,  # 增加该数值会报错
                            removesalt=True,
                            log=False,
                            fingerprints=True)
            # 将输出的csv文件中的SMILES由AUTOGEN_result_{}转换为正常输入的SMILES
            df = pd.read_csv(fingerprint_output_file)
            df.drop('Name', axis=1)
            df.insert(0, self.index_name, pd.Series(SMILES))
            df.to_csv(fingerprint_output_file, index=False)
        self.MergeResult()

    def MergeResult(self, result_csv='merged_FP.csv'):
        """
        用于将分次计算的特征文件合并成一个csv文件保存
        :param result_csv: 保存的合并文件名
        """
        csv_files = [file for file in os.listdir(self.save_dir) if file.endswith(".csv")]
        # 非首次运行时会将合并文件也读取进来，需要进行排除
        if result_csv in csv_files:
            csv_files.remove(result_csv)
        # 存储每个特征文件的dataframe
        dataframes = []
        for csv_file in csv_files:
            df = pd.read_csv(os.path.join(self.save_dir, csv_file), index_col=self.index_name)
            dataframes.append(df)
        logger.info("合并分子指纹数据...")
        # 合并所有Dataframe
        merged_df = pd.concat(dataframes, axis=1, sort=False)
        merged_df.to_csv(os.path.join(self.save_dir, result_csv))
        logger.info("Dataframe shape: " + str(merged_df.shape))
