# global variables
filetime = "20221221"
# 原始的集成数据集
workbookpath = f"./result/{filetime}/数据表汇总.xlsx"
# 从原始数据集中挑选出脑部与血液浓度的数据集
raw_csvfilepath = f"./result/{filetime}/BrainBlood.csv"
# 计算得到最大脑血比的数据集
ratio_csvfilepath = f"./result/{filetime}/MaxBrainBloodRatio.csv"
# 计算出药物的Mordred描述符以及最大脑血比的数据集
desc_csvfilepath = f"./result/{filetime}/RatioDescriptors.csv"
MACCS_csvfilepath = f"./result/{filetime}/RatioMACCSDescriptors.csv"
ECCF_csvfilepath = f"./result/{filetime}/RatioECCFDescriptors.csv"

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
            "n_estimators": 650,
            "learning_rate": 0.025,
            'max_depth': 12,
            'lambda': 0.28343954855304093,
            'alpha': 0.3962717638556472,
            'min_child_weight': 1,
            'gamma': 0,
            'colsample_bytree': 0.7,
            'colsample_bylevel': 0.9,
            'colsample_bynode': 0.6,
        },
        'brain_params': {
            # 'n_estimators': 1550,
            # 'learning_rate': 0.018,
            # 'max_depth': 24,
            # 'lambda': 0.07683686528439758,
            # 'alpha': 0.008538159369120378,
            # 'min_child_weight': 16,
            # 'gamma': 8,
            # 'colsample_bytree': 0.4,
            # 'colsample_bylevel': 0.8,
            # 'colsample_bynode': 0.4,
            # 'n_estimators': 1750,
            # 'learning_rate': 0.025,
            # 'max_depth': 25,
            # 'lambda': 0.5013166050987867,
            # 'alpha': 0.009790328397243015,
            # 'min_child_weight': 16,
            # 'gamma': 18,
            # 'colsample_bytree': 0.8,
            # 'colsample_bylevel': 1.0,
            # 'colsample_bynode': 0.6000000000000001,
            'n_estimators': 2200,
            'learning_rate': 0.02,
            'max_depth': 5,
            'lambda': 0.010075781713370716,
            'alpha': 0.19312610292731117,
            'min_child_weight': 8,
            'gamma': 17,
            'colsample_bytree': 1.0,
            'colsample_bylevel': 0.5,
            'colsample_bynode': 0.9,
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
        }
    },
    'SVM': {
        'blood_params': {
            'C': 10,
            'gamma': 'scale',
            'tol': 0.01,
            'max_iter': 10000,
            'epsilon': 0.9160690029281867,
        },
        'brain_params': {
            'C': 10,
            'gamma': 'auto',
            'tol': 0.0001,
            'max_iter': 1000,
            'epsilon': 0.39178737601316815,
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
        }
    }
}

# 特征筛选
# blood_fea = [162, 222, 254, 255, 261, 300, 320, 325, 338, 369, 396, 441, 446, 474, 481, 489, 502, 514, 529,
# 530, 541, 549, 565, 568, 570, 582, 594, 598, 602, 631, 632, 638, 645, 646, 648, 802, 807, 832, 986, 1145, 1226,
# 1232, 1266, 1287, 1289, 1297, 1316, 1356, 1539, 1544]

# blood_fea = [5, 162, 320, 338, 396, 453, 481, 489, 502, 529, 540, 565, 568, 600, 632, 646, 788, 802, 838, 1078,
# 1148, 1165, 1226, 1232, 1241, 1260, 1281, 1287, 1309, 1543]

blood_fea = [8, 85, 131, 132, 148, 162, 220, 247, 264, 311, 338, 369, 396, 441, 453, 481, 489, 497, 529, 568, 597, 598,
             602, 610, 618, 644, 645, 792, 802, 815, 847, 903, 1064, 1069, 1134, 1140, 1213, 1226, 1232, 1246, 1262,
             1297, 1301, 1306, 1330, 1336, 1383, 1539, 2071, 2469]

# brain_fea = [3, 40, 150, 164, 243, 246, 254, 255, 261, 310, 342, 368, 369, 449, 450, 458, 497, 506, 529, 542, 549,
# 578, 602, 604, 610, 618, 637, 642, 644, 646, 770, 781, 801, 814, 846, 986, 999, 1065, 1078, 1136, 1143, 1157, 1278,
# 1316, 1329, 1330, 1336, 1543, 1545, 1547]

# brain_fea = [3, 115, 184, 243, 261, 310, 342, 368, 369, 449, 450, 458, 506, 546, 610, 781, 791, 824, 832, 999, 1065, 1143, 1157, 1281, 1316, 1339, 1529, 1539, 1541, 1545]
brain_fea = [3, 67, 78, 186, 243, 264, 270, 300, 334, 341, 359, 397, 449, 450, 457, 458, 502, 506, 542, 546, 565, 570,
             572, 594, 602, 604, 610, 618, 630, 642, 648, 651, 678, 729, 755, 781, 792, 815, 986, 1060, 1136, 1139,
             1143, 1163, 1226, 1246, 1278, 1539, 1552, 2544]
X_fea = [3, 5, 8, 10, 11, 38, 41, 50, 62, 78, 84, 123, 135, 136, 140, 143, 144, 145, 146, 148, 151, 163, 168, 186, 187,
         194, 203, 205, 213, 214, 220, 221, 222, 230, 231, 232, 243, 252, 254, 255, 261, 270, 279, 289, 310, 311, 319,
         323, 333, 337, 338, 341, 342, 359, 368, 369, 396, 397, 404, 405, 414, 423, 441, 449, 450, 452, 453, 457, 458,
         465, 466, 481, 489, 497, 506, 529, 530, 534, 540, 541, 542, 546, 548, 549, 561, 565, 566, 568, 570, 572, 574,
         577, 578, 581, 582, 598, 599, 610, 626, 630, 631, 637, 638, 642, 644, 645, 646, 648, 665, 755, 780, 781, 803,
         807, 825, 827, 831, 836, 846, 848, 906, 981, 986, 993, 999, 1003, 1057, 1059, 1060, 1065, 1078, 1084, 1086,
         1134, 1136, 1139, 1140, 1143, 1144, 1157, 1161, 1162, 1165, 1209, 1211, 1212, 1215, 1219, 1221, 1226, 1228,
         1232, 1233, 1235, 1236, 1239, 1245, 1247, 1248, 1252, 1253, 1256, 1257, 1260, 1264, 1278, 1286, 1287, 1299,
         1302, 1316, 1317, 1318, 1319, 1320, 1325, 1328, 1329, 1330, 1339, 1352, 1354, 1355, 1356, 1357, 1358, 1360,
         1362, 1365, 1369, 1436, 1444, 1448, 1524, 1534, 1538, 1543, 1544, 1545, 1575]
