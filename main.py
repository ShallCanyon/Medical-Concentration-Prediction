import numpy as np

import global_config as cfg
import os
from model_training import train_model
from data_preprocess import *


def main():
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
        calculate_desc(ratio_csvfilepath, desc_csvfilepath,
                       Mordred=True, MACCS=False, ECFP=False)

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

        r2_list = []
        rmse_list = []
        val_r2_list = []
        val_rmse_list = []
        for i in range(10):
            training_result = train_model(ratio_X, ratio_y, cfg.model_type, param_name='ratio_params')
            print(f"Ratio data of epoch {i + 1}:")
            print("\tR2 Scores: %0.4f (+/- %0.2f)" %
                  (training_result.get("R2").mean(), training_result.get("R2").std()))
            r2_list.append(training_result.get("R2").mean())
            print("\tRMSE Scores: %0.4f (+/- %0.2f)" %
                  (training_result.get("RMSE").mean(), training_result.get("RMSE").std()))
            rmse_list.append(training_result.get("RMSE").mean())

            # print("\tNRMSE Scores: %0.4f (+/- %0.2f)" %
            #       (training_result.get("NRMSE").mean(), training_result.get("NRMSE").std()))
            print("Validation: ")
            print("\tR2 Scores: %0.4f" % training_result.get("Val_R2"))
            val_r2_list.append(training_result.get("Val_R2"))
            print("\tRMSE Scores: %0.4f" % training_result.get("Val_RMSE"))
            val_rmse_list.append(training_result.get("Val_RMSE"))
            # print("\tNRMSE Scores: %0.4f" % training_result.get("Val_NRMSE"))
        print("\nSummery:")
        print("\tTotal R2 Scores: %0.3f" % np.array(r2_list).mean())
        print("\tTotal RMSE Scores: %0.3f" % np.array(rmse_list).mean())
        print("\tTotal Validation R2 Scores: %0.3f" % np.array(val_r2_list).mean())
        print("\tTotal Validation RMSE Scores: %0.3f" % np.array(val_rmse_list).mean())


if __name__ == '__main__':
    # main()
    # raw_csvfilepath = cfg.raw_csvfilepath
    #
    # brain_csv = f"./result/{cfg.filetime}/MaxBrain.csv"
    # select_max_organ_data(raw_csvfilepath, brain_csv, 'brain')

    organ_name = 'brain'
    certain_time = 30
    organ_csv = f"./result/{cfg.filetime}/{organ_name}-{certain_time}min.csv"
    select_certain_time_organ_data(cfg.workbookpath, organ_csv, organ_name, certain_time)
