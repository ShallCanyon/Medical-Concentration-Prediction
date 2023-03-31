from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import VarianceThreshold, RFE,  mutual_info_classif, \
    mutual_info_regression, SelectPercentile, SelectFromModel
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression as LR
from time import time
import global_config as cfg
import pandas as pd
import numpy as np

logger = cfg.logger


class FeatureExtraction:
    """
    用于筛选化合物特征（描述符）的类
    """
    def __init__(self, X, y, mode='regression', VT_threshold=0.02, RFE_features_to_select=50, UFE_percentile=80):
        """
        :param X: 输入的特征数据
        :param y: 输入的标签数据
        :param mode: 选择回归数据或者分类数据，可选项为：'regression', 'classification'
        :param VT_threshold: VarianceThreshold的阈值
        :param RFE_features_to_select: RFE筛选的最终特征数
        :param UFE_percentile: UFE筛选的百分比
        """
        self.X = X
        self.y = y
        if mode not in ['regression', 'classification']:
            raise ValueError("Mode should be 'regression' or 'classification'")
        self.mode = mode
        self.VT_threshold = VT_threshold
        self.RFE_features_to_select = RFE_features_to_select
        self.UFE_percentile = UFE_percentile

    def get_VT(self):
        # deleted all features that were either one or zero in more than 98% of samples
        selector = VarianceThreshold(self.VT_threshold)
        return selector

    def get_RFE(self):
        global RF
        if self.mode == 'regression':
            from sklearn.ensemble import RandomForestRegressor as RF
        if self.mode == 'classification':
            from sklearn.ensemble import RandomForestClassifier as RF
        # base estimator SVM
        # estimator = SVC(kernel="rbf")
        # estimator = LR(max_iter=10000, solver='liblinear', class_weight='balanced')
        estimator = RF(n_jobs=-1, verbose=False)
        selector = RFE(estimator=estimator, n_features_to_select=self.RFE_features_to_select, verbose=False)
        # selector = RFECV(estimator=estimator, step=1, cv=StratifiedKFold(2),
        #           scoring='accuracy', n_jobs=-1)
        return selector

    def get_UFE(self):
        selector = None
        if self.mode == 'regression':
            selector = SelectPercentile(score_func=mutual_info_regression, percentile=self.UFE_percentile)
        if self.mode == 'classification':
            selector = SelectPercentile(score_func=mutual_info_classif, percentile=self.UFE_percentile)
        return selector

    def tree_based_selection(self, X, y):
        if self.mode == 'regression':
            clf = ExtraTreesRegressor()
        if self.mode == 'classification':
            clf = ExtraTreesClassifier()
        clf = clf.fit(X, y)
        model = SelectFromModel(clf, prefit=True)
        X_new = model.transform(X)
        return X_new

    def feature_extraction(self, VT=True, TBE=True, UFE=True, RFE=True):
        """
        连续调用特征筛选方法筛选特征
        :param VT: 是否启用VT方法
        :param TBE: 是否启用TBE方法
        :param UFE: 是否启用UFE方法
        :param RFE: 是否启用RFE方法
        :return: 完成筛选的特征
        """
        X = self.X
        if VT:
            X = self.get_VT().fit_transform(X, self.y)
            logger.info(f"X shape after Variance Threshold: {X.shape}")
        if TBE:
            X = self.tree_based_selection(X, self.y)
            logger.info(f"X shape after Tree Based Selection: {X.shape}")
        if UFE:
            X = self.get_UFE().fit_transform(X, self.y)
            logger.info(f"X shape after Select Percentile: {X.shape}")
        if RFE:
            X = self.get_RFE().fit_transform(X, self.y)
            logger.info(f"X shape after Recursive Feature Elimination: {X.shape}")

        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
