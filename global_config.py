# global variables
filetime = "20221221"
# 原始的集成数据集
workbookpath = f"./result/{filetime}/数据表汇总{filetime}.xlsx"
# 从原始数据集中挑选出脑部与血液浓度的数据集
raw_csvfilepath = f"./result/{filetime}/BrainBlood.csv"
# 计算得到最大脑血比的数据集
ratio_csvfilepath = f"./result/{filetime}/MaxBrainBloodRatio.csv"
# 计算出药物的Mordred描述符以及最大脑血比的数据集
desc_csvfilepath = f"./result/{filetime}/RatioDescriptors.csv"
MACCS_csvfilepath = f"./result/{filetime}/RatioMACCSDescriptors.csv"
ECFP_csvfilepath = f"./result/{filetime}/RatioECFPDescriptors.csv"
padel_csvfilepath = f"./result/{filetime}/PadelDescriptors.csv"

model_enum = ['XGB', 'LGBM', 'SVM', 'RF', 'MLP']
model_type = model_enum[0]

"""SVM
Blood data:
R2 Scores: 0.0415 (+/- 0.08)
RMSE Scores: 5.6040 (+/- 3.33)
Brain data:
R2 Scores: 0.2458 (+/- 0.32)
RMSE Scores: 55.0709 (+/- 46.54)
"""

"""RF
Blood data:
R2 Scores: 0.4831 (+/- 0.31)
RMSE Scores: 4.2956 (+/- 1.63)
Brain data:
R2 Scores: -7.5106 (+/- 10.84)
RMSE Scores: 53.1822 (+/- 33.33)
"""
model_params = {
    'XGB': {
        'blood_params': {
            # 'n_estimators': 1950,
            # 'learning_rate': 0.014,
            # 'max_depth': 18,
            # 'lambda': 8.645496158267079,
            # 'alpha': 0.45639661861114994,
            # 'min_child_weight': 1,
            # 'gamma': 9,
            # 'colsample_bytree': 0.30000000000000004,
            # 'colsample_bylevel': 0.30000000000000004,
            # 'colsample_bynode': 0.9,
            # 'n_estimators': 2200,
            # 'learning_rate': 0.028,
            # 'max_depth': 29,
            # 'lambda': 1.7820234285458951,
            # 'alpha': 0.0019187137167321823,
            # 'min_child_weight': 1,
            # 'gamma': 1,
            # 'colsample_bytree': 0.2,
            # 'colsample_bylevel': 0.9,
            # 'colsample_bynode': 0.7000000000000001,
            'n_estimators': 1700,
            'learning_rate': 0.026,
            'max_depth': 26,
            'lambda': 0.0022106369528429484,
            'alpha': 0.9133162515639958,
            'min_child_weight': 18,
            'gamma': 9,
            'colsample_bytree': 0.9,
            'colsample_bylevel': 0.6,
            'colsample_bynode': 0.3,
        },
        'brain_params': {
            'n_estimators': 1450,
            'learning_rate': 0.013,
            'max_depth': 11,
            'lambda': 0.1245428483067459,
            'alpha': 0.20833503659100544,
            'min_child_weight': 1,
            'gamma': 12,
            'colsample_bytree': 0.4,
            'colsample_bylevel': 0.1,
            'colsample_bynode': 0.4,
        },
        'ratio_params': {
            'n_estimators': 1400,
            'learning_rate': 0.018,
            'max_depth': 23,
            'lambda': 5.599908028889678,
            'alpha': 8.503428132294024,
            'min_child_weight': 11,
            'gamma': 3,
            'colsample_bytree': 0.4,
            'colsample_bylevel': 1.0,
            'colsample_bynode': 0.3,
        }
    },
    'LGBM': {
        'blood_params': {
            # 'boosting_type': 'gbdt',
            # 'max_depth': 25,
            # 'learning_rate': 0.023,
            # 'n_estimators': 1650,
            # 'objective': 'regression',
            # 'min_child_samples': 8,
            # 'reg_lambda': 0.023612179335686056,
            # 'reg_alpha': 8.209581280307875,
            'boosting_type': 'gbdt',
            'max_depth': 27,
            'learning_rate': 0.03,
            'n_estimators': 2350,
            'objective': 'regression',
            'min_child_samples': 19,
            'reg_lambda': 1.5080500800010115,
            'reg_alpha': 0.008779034615465546,
        },
        'brain_params': {
            # 'boosting_type': 'dart',
            # 'max_depth': 2,
            # 'learning_rate': 0.028,
            # 'n_estimators': 2900,
            # 'objective': 'regression',
            # 'min_child_samples': 27,
            # 'reg_lambda': 3.99101992359789,
            # 'reg_alpha': 0.00690882557106338,
            'boosting_type': 'dart',
            'max_depth': 22,
            'learning_rate': 0.023,
            'n_estimators': 2350,
            'objective': 'regression',
            'min_child_samples': 5,
            'reg_lambda': 2.818010179932352,
            'reg_alpha': 0.06121129321041584,
        },
        'ratio_params': {
            'boosting_type': 'dart',
            'max_depth': 9,
            'learning_rate': 0.014,
            'n_estimators': 2900,
            'objective': 'regression',
            'min_child_samples': 30,
            'reg_lambda': 0.01837328293363213,
            'reg_alpha': 0.028586571129442278,
        }
    },
    'SVM': {
        'blood_params': {
            'C': 8.91163936981316,
            'gamma': 'scale',
            'tol': 0.0001,
            'max_iter': 1000,
            'epsilon': 0.5292101661636676,
        },
        'brain_params': {
            'C': 3.025643003884724,
            'gamma': 'scale',
            'tol': 0.0001,
            'max_iter': 5000,
            'epsilon': 0.7640969657803434,
        },
        'ratio_params': {
            'C': 8.911791674163041,
            'gamma': 'scale',
            'tol': 0.001,
            'max_iter': 10000,
            'epsilon': 0.779120493479141,
        }
    },
    'RF': {
        'blood_params': {
            'n_estimators': 2000,
            'max_depth': 7,
        },
        'brain_params': {
            'n_estimators': 2800,
            'max_depth': 27,
        },
        'ratio_params': {

        }
    },
    'MLP': {
        'blood_params': {
            'hidden_layer_sizes': (50,),
            'activation': 'relu',
            'solver': 'adam',
            'max_iter': 5000,
        },
        'brain_params': {
            'hidden_layer_sizes': (200,),
            'activation': 'relu',
            'solver': 'adam',
            'max_iter': 5000,
        },
        'ratio_params': {

        }
    }
}

# 特征筛选
# blood_fea = [5, 162, 320, 338, 396, 453, 481, 489, 502, 529, 540, 565, 568, 600, 632, 646, 788, 802, 838,
# 1078, 1148, 1165, 1226, 1232, 1241, 1260, 1281, 1287, 1309, 1543] blood_fea = [12, 105, 148, 320, 348, 369, 403,
# 462, 758, 851, 1041, 1058, 1200, 1317, 1407, 1436, 1755, 1954, 2143, 2166, 2268, 2314, 2360, 2442, 2959, 2966,
# 3075, 3152, 3260, 3269, 3476, 3760]
# blood_fea = [25, 33, 38, 49, 50, 60, 82, 85, 90, 94, 96, 98, 109, 122, 124, 126, 146, 149, 156, 176, 185, 223, 225, 226, 315, 316, 317, 319, 324, 365]
blood_fea = [51, 101, 108, 133, 140, 166, 193, 209, 217, 233, 234, 238, 250, 261, 262, 278, 282, 283, 289, 293, 298, 306, 353, 635, 638, 709, 820, 822, 849, 853, 857, 860, 862, 877, 878, 903, 922, 965, 968, 969, 971, 1043, 1061, 1063, 1065, 1090, 1113, 1250, 1865, 2078]

# brain_fea = [3, 40, 150, 164, 243, 246, 254, 255, 261, 310, 342, 368, 369, 449, 450, 458, 497, 506, 529, 542, 549,
# 578, 602, 604, 610, 618, 637, 642, 644, 646, 770, 781, 801, 814, 846, 986, 999, 1065, 1078, 1136, 1143, 1157, 1278,
# 1316, 1329, 1330, 1336, 1543, 1545, 1547]

# brain_fea = [3, 115, 184, 243, 261, 310, 342, 368, 369, 449, 450, 458, 506, 546, 610, 781, 791, 824, 832, 999,
# 1065, 1143, 1157, 1281, 1316, 1339, 1529, 1539, 1541, 1545] brain_fea = [3, 67, 78, 186, 243, 264, 270, 300, 334,
# 341, 359, 397, 449, 450, 457, 458, 502, 506, 542, 546, 565, 570, 572, 594, 602, 604, 610, 618, 630, 642, 648, 651,
# 678, 729, 755, 781, 792, 815, 986, 1060, 1136, 1139, 1143, 1163, 1226, 1246, 1278, 1539, 1552, 2544] brain_fea = [
# 98, 205, 342, 356, 562, 572, 679, 693, 727, 926, 1078, 1082, 1190, 1209, 1216, 1222, 1265, 1380, 1400, 1583, 1943,
# 1986, 2104, 2138, 2209, 2287, 2543, 2667, 2691, 2751, 2995, 3167, 3331, 3403, 3405, 3435, 3472, 3515, 3666, 3671,
# 3680, 3772]
# brain_fea = [1, 2, 17, 82, 98, 100, 106, 122, 124, 129, 156, 163, 203, 220, 226, 299, 303, 312, 313, 314, 317, 320, 350, 352, 362, 461, 903, 1003, 1188, 1242]
brain_fea = [2, 25, 26, 27, 28, 64, 69, 93, 109, 166, 167, 193, 209, 217, 233, 238, 258, 273, 274, 280, 282, 289, 294, 306, 540, 638, 704, 768, 837, 842, 849, 851, 865, 875, 909, 948, 949, 967, 1051, 1054, 1055, 1056, 1065, 1066, 1113, 1318, 1393, 1678, 1754, 1820]
# X_fea = [17, 60, 77, 82, 85, 86, 93, 98, 106, 108, 109, 124, 144, 154, 163, 171, 176, 221, 303, 314, 315, 316, 317,
# 320, 323, 446, 456, 982, 1213, 1439]
X_fea = [26, 82, 133, 140, 152, 175, 201, 204, 218, 246, 261, 262, 269, 280, 286, 351, 352, 484, 507, 591, 592, 618, 629, 632, 641, 728, 750, 763, 769, 820, 822, 832, 838, 842, 845, 851, 855, 863, 865, 893, 933, 966, 969, 1062, 1063, 1065, 1067, 1077, 1089, 3290]