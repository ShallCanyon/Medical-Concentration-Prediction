import os
import time
import deepchem as dc
import numpy as np
import torch

import global_config as cfg
import DataLogger
from deepchem import feat, data, metrics, hyper
from deepchem.models import AttentiveFPModel, optimizers, TorchModel, GCNModel
from sklearn.model_selection import train_test_split
from data_preprocess import get_X_Y_by_ratio, get_SMILE_Y

# global settings
cur_time = time.localtime()
parent_folder = f"./Models/{time.strftime('%Y%m%d', cur_time)}"
save_folder = f"{parent_folder}/{time.strftime('%H%M%S', cur_time)}"
if not os.path.exists(parent_folder):
    os.mkdir(parent_folder)
if not os.path.exists(save_folder):
    os.mkdir(save_folder)
logger = DataLogger.DataLogger(f"{save_folder}/logger.txt").getlog()


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
        num_layers = model_params['num_layers']
        dropout = model_params['dropout']
        lr = model_params['learning_rate']
        # number_atom_features = model_params['number_atom_features']
        graph_feat_size = model_params['graph_feat_size']
        # number_bond_features = model_params['number_bond_features']
        batch_size = model_params['batch_size']
        num_timesteps = model_params['num_timesteps']

        learning_rate = optimizers.ExponentialDecay(lr, 0.9, 200)
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
        "num_layers": [3],
        "dropout": [0.08],
        "learning_rate": [0.001],
        # "number_atom_features": [10, 20, 30, 40, 50],
        "graph_feat_size": [500],
        # "number_bond_features": [10, 12, 14, 16, 18, 20],
        "batch_size": [16],
        "num_timesteps": [2, 3, 4, 5, 6]
    }
    best_model, best_hyperparams, all_results = \
        optimizer.hyperparam_search(params_dict, train_dataset, valid_dataset, metric, max_iter=2)
    return best_model, best_hyperparams, all_results


def getMetaLearner(large_datasets):
    # learner = metalearning.MetaLearner()
    # learner.compute_model(inputs=large_datasets, variables=None, training=True)
    # return learner
    pass


# def FS_Learning(large_datasets):
#     learner = getMetaLearner(large_datasets)
#     maml = metalearning.MAML(learner=learner,
#                              learning_rate=0.001,
#                              optimization_steps=1,
#                              meta_batch_size=10,
#                              model_dir="./metalearning")
#     maml.fit(steps=10)
#     maml.train_on_current_task()


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

    learning_rate = optimizers.ExponentialDecay(lr, 0.9, 200)
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
    best_model = AttentiveFPModel(mode='regression', num_layers=3, n_tasks=1, dropout=0.08,
                                  graph_feat_size=500, num_timesteps=2, number_atom_features=30,
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


def train_models(X, y, epoch=5, tuning_mode=False):
    lr = [0.003, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    # Ratio
    for i in range(epoch):
        train_dataset, test_dataset, valid_dataset = get_datasets(X=X, y=y)
        model_dir = f"{save_folder}/model{i}"

        # splitter = dc.splits.RandomSplitter()
        # # print(type(train_dataset))
        # cv_train = splitter.k_fold_split(train_dataset, 5)
        # print(cv_train)

        if not tuning_mode:
            train_scores, valid_scores, model, metric_list = \
                train_GCNModel(train_dataset, valid_dataset, lr=lr[1], model_dir=model_dir)
            test_scores = model.evaluate(test_dataset, metric_list)
            logger.info(f"Model{i} result:")
            logger.info(f"Learning rate:{lr[1]}")
            logger.info(f"Training: {train_scores}")
            logger.info(f"Valid: {valid_scores}")
            logger.info(f"Test: {test_scores}\n")
        else:
            best_model, best_hyperparams, all_results = param_tuning(train_dataset, valid_dataset)
            logger.info(best_model)
            logger.info(best_hyperparams)
            logger.info(all_results)

    # # Blood
    # train_scores, valid_scores = train_GCNModel(*get_datasets(X=X, y=blood_y))
    # print("Blood result:")
    # print(f"Training: {train_scores}")
    # print(f"Valid: {valid_scores}")
    #
    # # Brain
    # train_scores, valid_scores = train_GCNModel(*get_datasets(X=X, y=brain_y))
    # print("Brain result:")
    # print(f"Training: {train_scores}")
    # print(f"Valid: {valid_scores}")


if __name__ == '__main__':
    # tf.config.set_soft_device_placement(True)
    # tf.debugging.set_log_device_placement(True)
    #
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    # tf.config.experimental.set_memory_growth(gpus[0], True)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    """ 预测比值
    _, blood_y, brain_y, ratio_y, SMILES = get_X_Y_by_ratio(cfg.padel_csvfilepath)
    X = featurize(SMILES, is_torch=True)
    # y = brain_y
    train_models(X, ratio_y, epoch=5, tuning_mode=False)
    """

    """ 预测最大脑部
    brain_csv = f"./result/{cfg.filetime}/MaxBrain.csv"
    SMILES, y = get_SMILE_Y(brain_csv)

    X = featurize(SMILES, is_torch=True)
    train_models(X, y, epoch=5, tuning_mode=False)
    """

    brain_csv = f"./result/{cfg.filetime}/MaxBrain.csv"
    SMILES, y = get_SMILE_Y(brain_csv)

    X = featurize(SMILES, is_torch=True)
    train_models(X, y, epoch=5, tuning_mode=False)

