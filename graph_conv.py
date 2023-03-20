import os
import time
import deepchem as dc
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

import data_preprocess
import global_config as cfg
from deepchem import feat, data, metrics, hyper
from deepchem.models import AttentiveFPModel, optimizers, TorchModel, GCNModel
from sklearn.model_selection import train_test_split
from data_preprocess import get_X_Y_by_ratio, get_SMILE_Y, split_null_from_csv


logger = cfg.logger


def featurize(SMILES, is_torch=False):
    if is_torch:
        # print("Creating torch-like featurizer...")
        logger.info("Creating torch-like featurizer...")
        featurizer = feat.MolGraphConvFeaturizer(use_edges=True)
    else:
        # print("Create keras-like featurizer...")
        logger.info("Creating keras-like featurizer...")
        featurizer = feat.ConvMolFeaturizer(master_atom=False,
                                            per_atom_fragmentation=False)
    data = featurizer.featurize(SMILES, log_every_n=50)
    return data


def get_datasets(X, y, test_size=0.1, valid_size=0.1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=valid_size)
    train_dataset = data.NumpyDataset(X=X_train, y=y_train)
    test_dataset = data.NumpyDataset(X=X_test, y=y_test)
    valid_dataset = data.NumpyDataset(X=X_val, y=y_val)
    return train_dataset, test_dataset, valid_dataset


def param_tuning(train_dataset, valid_dataset):
    def model_builder(**model_params):
        num_layers = model_params.get('num_layers', 2)
        dropout = model_params.get('dropout', 0.0)
        lr = model_params.get('learning_rate', 0.002)
        graph_feat_size = model_params.get('graph_feat_size', 200)
        batch_size = model_params.get('batch_size', 16)
        num_timesteps = model_params.get('num_timesteps', 2)

        # learning_rate = optimizers.ExponentialDecay(lr, 0.9, 200)
        learning_rate = optimizers.ExponentialDecay(lr, 0.9, 500)
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        model = AttentiveFPModel(mode='regression', n_tasks=1, num_timesteps=num_timesteps,
                                 dropout=dropout,
                                 num_layers=num_layers,
                                 graph_feat_size=graph_feat_size,
                                 # number_atom_features=number_atom_features,
                                 # number_bond_features=number_bond_features,
                                 self_loop=True,
                                 batch_size=batch_size,
                                 # optimizer=optimizer,
                                 learning_rate=learning_rate,
                                 device='cuda')
        return model

    optimizer = hyper.GridHyperparamOpt(model_builder)
    metric = metrics.Metric(metrics.pearson_r2_score)
    params_dict = {
        # "num_layers": [2, 3, 4],
        # "dropout": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        # "learning_rate": [0.001, 0.002, 0.01, 0.02],
        # "graph_feat_size": [200, 300, 500, 700, 1000],
        # "batch_size": [16, 32, 64, 128],
        # "num_timesteps": [2, 3, 4],

        "num_layers": [2],
        "dropout": [0.0],
        "learning_rate": [0.002],
        "graph_feat_size": [700],
        "batch_size": [16],
        "num_timesteps": [3]
    }
    best_model, best_hyperparams, all_results = \
        optimizer.hyperparam_search(params_dict, train_dataset, valid_dataset, metric, max_iter=2)
    return best_model, best_hyperparams, all_results


def custom_metric_func(digit):
    return np.sqrt(np.mean(digit))


def train_GCNModel(train_dataset, valid_dataset, nb_epoch=300, lr=0.001, model_dir=None):
    # Metrics initialization
    # pearson_metric = metrics.Metric(metrics.pearson_r2_score, np.mean, mode='regression')
    r2_metric = metrics.Metric(metrics.r2_score, np.mean, mode='regression')
    mse_metric = metrics.Metric(metrics.mean_squared_error, custom_metric_func, mode='regression')

    logger.info("Start training Model...")
    # model = GCNModel(mode='regression', n_tasks=1, batch_size=16, learning_rate=0.001, device='cuda')
    # model = AttentiveFPModel(mode='regression', num_layers=4, n_tasks=1, dropout=0.0,
    #                          graph_feat_size=200, num_timesteps=2, number_atom_features=30,
    #                          number_bond_features=11, self_loop=True,
    #                          batch_size=16, learning_rate=0.001, device='cuda')
    # loss = model.fit(train_dataset, nb_epoch=10)
    # print("Ratio loss:", loss)
    # train_scores = model.evaluate(train_dataset, [metric])
    # # print(train_scores)
    # # assert train_scores['mean-pearson_r2_score'] > 0.5, train_scores
    #
    # valid_scores = model.evaluate(valid_dataset, [metric])
    # # print(train_scores)
    # # assert valid_scores['mean-pearson_r2_score'] > 0.3, valid_scores
    # return train_scores, valid_scores

    # best_model, best_hyperparams, all_results = param_tuning(train_dataset, valid_dataset)
    # print("Best_hyperparams: ", best_hyperparams)

    # learning_rate = optimizers.ExponentialDecay(lr, 0.9, 200)
    learning_rate = optimizers.ExponentialDecay(0.002, 0.9, 200)
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    # best_model = GCNModel(mode='regression',
    #                       n_tasks=1,
    #                       graph_conv_layers=[128, 128],
    #                       activation=torch.nn.Softsign(),
    #                       residual=True,
    #                       dropout=0.5,
    #                       # predictor_dropout=0.0,
    #                       # predictor_hidden_feats=128,
    #                       number_atom_features=30,
    #                       batchnorm=True,
    #                       batch_size=64,
    #                       optimizer=optimizer,
    #                       device='cuda',
    #                       model_dir=model_dir)
    """param of ratio"""
    # best_model = AttentiveFPModel(mode='regression', num_layers=2, n_tasks=1, dropout=0.1,
    #                               graph_feat_size=200, num_timesteps=2, number_atom_features=30,
    #                               number_bond_features=11, self_loop=True,
    #                               batch_size=64, device='cuda',
    #                               learning_rate=learning_rate,
    #                               # optimizer=optimizer,
    #                               # loss=torch.nn.SmoothL1Loss(),
    #                               model_dir=model_dir)
    """param of brain"""
    # best_model = AttentiveFPModel(mode='regression', num_layers=3, n_tasks=1, dropout=0.08,
    #                               graph_feat_size=500, num_timesteps=2, number_atom_features=30,
    #                               number_bond_features=11, self_loop=True,
    #                               batch_size=16, device='cuda',
    #                               learning_rate=learning_rate,
    #                               # optimizer=optimizer,
    #                               # loss=torch.nn.SmoothL1Loss(),
    #                               model_dir=model_dir)
    """param of 60min brain"""
    best_model = AttentiveFPModel(mode='regression', num_layers=2, n_tasks=1, dropout=0.0,
                                  graph_feat_size=700, num_timesteps=3, number_atom_features=30,
                                  number_bond_features=11, self_loop=True,
                                  batch_size=16, device='cuda',
                                  learning_rate=learning_rate,
                                  # optimizer=optimizer,
                                  # loss=torch.nn.SmoothL1Loss(),
                                  model_dir=model_dir)
    callback = dc.models.ValidationCallback(valid_dataset, 100, [r2_metric])
    if isinstance(train_dataset, list):
        for train in train_dataset:
            loss = best_model.fit(train, nb_epoch=nb_epoch, callbacks=callback)
    else:
        loss = best_model.fit(train_dataset, nb_epoch=nb_epoch, callbacks=callback)
    metric_list = [r2_metric, mse_metric]
    train_scores = best_model.evaluate(train_dataset, metric_list)
    valid_scores = best_model.evaluate(valid_dataset, metric_list)
    return train_scores, valid_scores, best_model, metric_list


def predict_on_model(model_dir, dataset):
    pearson_metric = metrics.Metric(metrics.pearson_r2_score, np.mean, mode='regression')
    r2_metric = metrics.Metric(metrics.r2_score, np.mean, mode='regression')
    mse_metric = metrics.Metric(metrics.mean_squared_error, custom_metric_func, mode='regression')
    metric_list = [pearson_metric, r2_metric, mse_metric]

    model = AttentiveFPModel(mode='regression', n_tasks=1, model_dir=model_dir)
    model.restore()
    scores = model.evaluate(dataset, metric_list)
    return scores


def train_AFP_model(X, y, epoch=5, tuning_mode=False):
    lr = [0.003, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    best_params = None
    if tuning_mode is True:
        train_dataset, test_dataset, valid_dataset = get_datasets(X=X, y=y)
        best_model, best_hyperparams, all_results = param_tuning(train_dataset, valid_dataset)
        # logger.info(best_model)
        logger.info(best_hyperparams)
        # logger.info(all_results)
        best_params = best_hyperparams
    else:
        for i in range(epoch):
            train_dataset, test_dataset, valid_dataset = get_datasets(X=X, y=y)
            model_dir = f"{cfg.model_save_folder}/model{i}"

            # splitter = dc.splits.RandomSplitter()
            # # print(type(train_dataset))
            # cv_train = splitter.k_fold_split(train_dataset, 5)
            # print(cv_train)
            train_scores, valid_scores, model, metric_list = \
                train_GCNModel(train_dataset, valid_dataset, lr=lr[1], model_dir=model_dir)
            test_scores = model.evaluate(test_dataset, metric_list)
            logger.info(f"Model{i} processed_data:")
            logger.info(f"Learning rate:{lr[1]}")
            logger.info(f"Training: {train_scores}")
            logger.info(f"Valid: {valid_scores}")
            logger.info(f"Test: {test_scores}\n")


if __name__ == '__main__':
    # tf.config.set_soft_device_placement(True)
    # tf.debugging.set_log_device_placement(True)
    #
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    # tf.config.experimental.set_memory_growth(gpus[0], True)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # 预测比值
    """ 
    _, blood_y, brain_y, ratio_y, SMILES = get_X_Y_by_ratio(cfg.padel_csvfilepath)
    X = featurize(SMILES, is_torch=True)
    # y = brain_y
    train_models(X, ratio_y, epoch=5, tuning_mode=False)
    """

    # 预测最大脑部
    """ 
    brain_csv = f"./processed_data/{cfg.filetime}/MaxBrain.csv"
    SMILES, y = get_SMILE_Y(brain_csv)

    X = featurize(SMILES, is_torch=True)
    train_models(X, y, epoch=5, tuning_mode=False)
    """

    # 预测30分脑部
    # organ_name = 'brain'
    # certain_time = 30
    # organ_csv = f"./processed_data/{cfg.filetime}/{organ_name}-{certain_time}min.csv"
    # data_df, empty_df = split_null_from_csv(organ_csv)
    # SMILES, y = get_SMILE_Y(data_df, y_column=f'{organ_name} mean{certain_time}min')
    #
    # X = featurize(SMILES, is_torch=True)
    # train_models(X, y, epoch=1, tuning_mode=True)

    # 预测60分脑部
    organ_name = 'brain'
    certain_time = 60
    organ_csv = f"./{cfg.parent_folder}/{cfg.filetime}/{organ_name}-{certain_time}min.csv"
    desc_csv = f"./{cfg.parent_folder}/{cfg.filetime}/{organ_name}-{certain_time}min-desc.csv"
    _, y, smiles = data_preprocess.read_single_column_data(desc_csv)
    X = featurize(smiles, is_torch=True)

    tuning = False
    logger.info("Model type: AttentiveFP")
    train_AFP_model(X, y, epoch=5, tuning_mode=tuning)

