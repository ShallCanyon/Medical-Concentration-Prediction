import numpy as np
import pandas as pd

import DataPreprocess
import global_config as cfg
import os
from model_training import train_model, model_training
from DataPreprocess import *
from params_tuning import ParamsTuning
from FeatureExtraction import FeatureExtraction

logger = cfg.logger


def ratio_predict():
    # 原始的集成数据集
    workbookpath = cfg.workbookpath
    # 从原始数据集中挑选出脑部与血液浓度的数据集
    raw_csvfilepath = cfg.raw_csvfilepath
    # 计算得到最大脑血比的数据集
    ratio_csvfilepath = cfg.ratio_csvfilepath
    # 计算出药物的Mordred描述符以及最大脑血比的数据集
    desc_csvfilepath = cfg.desc_csvfilepath
    MACCS_csvfilepath = cfg.MACCS_csvfilepath
    ECFP_csvfilepath = cfg.ECFP_csvfilepath
    generate_new_data = [False, False, False]

    print("Running...")
    if not os.path.exists(raw_csvfilepath) or generate_new_data[0]:
        print("Getting blood brain file...")
        get_brainblood_csv(workbookpath, raw_csvfilepath)

    if not os.path.exists(ratio_csvfilepath) or generate_new_data[1]:
        print("Calculating blood brain ratio...")
        calculate_blood_brain_ratio(raw_csvfilepath, ratio_csvfilepath)

    if not os.path.exists(desc_csvfilepath) or generate_new_data[2]:
        print("Calculating descriptors...")
        df = calculate_desc(ratio_csvfilepath, Mordred=True, MACCS=False, ECFP=False)
        df.to_csv(desc_csvfilepath, index=False)

    # calculate_desc(ratio_csvfilepath, ECCF_csvfilepath)
    start_training = True
    if start_training:
        X, blood_y, brain_y, ratio_y, SMILES = get_X_Y_by_ratio(cfg.padel_csvfilepath)
        feature_select = True
        if feature_select:
            # 特征筛选
            blood_X = X.iloc[:, cfg.blood_fea]
            brain_X = X.iloc[:, cfg.brain_fea]
            ratio_X = X.iloc[:, cfg.X_fea]
        else:
            blood_X = X
            brain_X = X
            ratio_X = X

        print("Start training model...")

        # training_result = train_model(blood_X, blood_y, cfg.model_type, param_name='blood_params')
        #
        # print("Blood data:")
        # print("\tR2 Scores: %0.4f (+/- %0.2f)" %
        #       (training_result.get("R2").mean(), training_result.get("R2").std()))
        # print("\tRMSE Scores: %0.4f (+/- %0.2f)" %
        #       (training_result.get("RMSE").mean(), training_result.get("RMSE").std()))
        # # print("\tNRMSE Scores: %0.4f (+/- %0.2f)" %
        # #       (training_result.get("NRMSE").mean(), training_result.get("NRMSE").std()))
        # print("Validation: ")
        # print("\tR2 Scores: %0.4f" % training_result.get("Val_R2"))
        # print("\tRMSE Scores: %0.4f" % training_result.get("Val_RMSE"))
        # # print("\tNRMSE Scores: %0.4f" % training_result.get("Val_NRMSE"))
        #
        # print()
        #
        # training_result = train_model(brain_X, brain_y, cfg.model_type, param_name='brain_params')
        # print("Brain data:")
        # print("\tR2 Scores: %0.4f (+/- %0.2f)" %
        #       (training_result.get("R2").mean(), training_result.get("R2").std()))
        # print("\tRMSE Scores: %0.4f (+/- %0.2f)" %
        #       (training_result.get("RMSE").mean(), training_result.get("RMSE").std()))
        # # print("\tNRMSE Scores: %0.4f (+/- %0.2f)" %
        # #       (training_result.get("NRMSE").mean(), training_result.get("NRMSE").std()))
        # print("Validation: ")
        # print("\tR2 Scores: %0.4f" % training_result.get("Val_R2"))
        # print("\tRMSE Scores: %0.4f" % training_result.get("Val_RMSE"))
        # # print("\tNRMSE Scores: %0.4f" % training_result.get("Val_NRMSE"))
        #
        # print()


if __name__ == '__main__':
    # ratio_predict()
    # raw_csvfilepath = cfg.raw_csvfilepath

    # brain_csv = f"./processed_data/{cfg.filetime}/MaxBrain.csv"
    # select_max_organ_data(raw_csvfilepath, brain_csv, 'brain')

    # 获取源数据
    organ_name = 'brain'
    certain_time = 60
    organ_csv = f"./{cfg.parent_folder}/{cfg.filetime}/{organ_name}-{certain_time}min.csv"
    desc_csv = f"./{cfg.parent_folder}/{cfg.filetime}/{organ_name}-{certain_time}min-desc.csv"
    if not os.path.exists(organ_csv):
        df = select_certain_time_organ_data(cfg.workbookpath, organ_name, certain_time)
        df.to_csv(organ_csv, encoding='utf-8')
    if not os.path.exists(desc_csv):
        data_df, empty_df = split_null_from_data(pd.read_csv(organ_csv))
        desc_df = calculate_desc(data_df)
        desc_df.to_csv(desc_csv, index=False)

    tune_mode = False
    feature_extract = True
    mode_train_times = 50
    # 数据预处理
    X, y, _ = get_single_column_data(desc_csv)
    sc = StandardScaler()
    X = pd.DataFrame(sc.fit_transform(X))
    logger.info(f"X shape: {X.shape}")
    if feature_extract:
        X = FeatureExtraction(X=X, y=y).feature_extraction(RFE=True)
        logger.info("Feature extracting...")

    model_type = cfg.model_enum[0]
    logger.info("Model type: " + model_type)
    # param_name = 'ratio_params'
    # model_params = cfg.model_params.get(model_type).get(param_name)

    if tune_mode:
        logger.info("Start model pre-tuning...")
        pt = ParamsTuning(model_type=model_type, n_trials=50, study_name="Tuning Study")
        study = pt.tune_params(X, y)
        logger.info("Best parameters: ", study.best_params)
        model = model_training(X, y, model_type=model_type, model_params=study.best_params, train_times=mode_train_times)
        # model.save_model()
    else:
        model_training(X, y, model_type=model_type, train_times=mode_train_times)
