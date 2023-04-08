import os
import random

import DataPreprocess
import global_config as cfg
import torch
import numpy as np
import learn2learn as l2l
from torch import nn
from learn2learn.data import MetaDataset, TaskDataset
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
from TorchNet import RegressionModel
from learn2learn.algorithms import MAML
from MedicalDatasets import MedicalDatasets
from DataLogger import DataLogger

logger = DataLogger(cfg.logger_filepath, 'MAML').getlog()


def read_tensor_datasets(device):
    base_dir = "./Datasets/"
    map = {}
    for path in os.listdir(base_dir):
        if path.endswith(".pt"):
            name = path.split("_")[0]
            map[name] = torch.load(base_dir + path)
    return map

"""
def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(batch, learner, criterion, adaptation_steps, shots, ways, device):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots * ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    # Adapt the model
    for step in range(adaptation_steps):
        train_loss = criterion(learner(adaptation_data), adaptation_labels)
        learner.adapt(train_loss)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    valid_loss = criterion(predictions, evaluation_labels)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_loss, valid_accuracy
"""


def main(ways=5,
         shots=1,
         model_lr=1e-3,
         maml_lr=0.5,
         meta_batch_size=32,
         adaptation_steps=1,
         num_iterations=600,
         seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Available device: {device}")

    # 从csv文件中读取并处理初始数据集
    csv_filepath = "D:\\ML\\Medical Data Process\\processed_data\\20230321\\OrganDataAt60min.csv"
    md = MedicalDatasets(csv_filepath)
    map = md.transform_organs_data(overwrite=True)
    md.save_dataframes2Tensor(map)

    # 将数据集处理成查询集与支持集
    target_organ = 'brain'
    torchDatasets = read_tensor_datasets(device)
    queryset = torchDatasets.pop(target_organ)
    supportset = torchDatasets
    # print(torchDatasets)

    meta_queryset = MetaDataset(queryset)
    meta_supportset = MetaDataset(ConcatDataset(supportset.values()))
    query_dataloader = DataLoader(meta_queryset, batch_size=8, shuffle=True)
    support_dataloader = DataLoader(meta_supportset, batch_size=meta_batch_size, shuffle=True)

    # transforms = [
    #     l2l.data.transforms.NWays(meta_dataset, n=5),
    #     l2l.data.transforms.KShots(meta_dataset, k=10),
    #     l2l.data.transforms.LoadData(meta_dataset),
    # ]
    # # taskset = TaskDataset(meta_dataset, transforms, num_tasks=len(torchDatasets))
    # taskset = TaskDataset(meta_dataset, transforms, num_tasks=len(meta_dataset))

    # 初始化模型
    # model = RegressionModel(input_size=map.get('brain').shape[1], n_hidden=32, output_size=1).to(device)
    model = RegressionModel(input_size=25, n_hidden=128, output_size=1).to(device)
    maml = MAML(model, lr=maml_lr)
    criterion = nn.MSELoss()
    opt = torch.optim.Adam(maml.parameters(), lr=model_lr)

    # 训练集（支持集）
    for iter, batch in enumerate(support_dataloader):  # num_tasks/batch_size
        opt.zero_grad()
        meta_valid_loss = 0.0

        # for each task in the batch
        effective_batch_size = batch[0].shape[0]
        for i in range(effective_batch_size):
            learner = maml.clone()

            # divide the data into support and query sets
            train_inputs, train_targets = batch[0][i].float(), batch[1][i].float()
            # x_support, y_support = train_inputs[::2], train_targets[::2]
            # x_query, y_query = train_inputs[1::2], train_targets[1::2]
            x_support, y_support = train_inputs[::2], train_targets
            x_query, y_query = train_inputs[1::2], train_targets

            for _ in range(adaptation_steps):  # adaptation_steps
                support_preds = learner(x_support)
                support_loss = criterion(support_preds, y_support)
                learner.adapt(support_loss)

            query_preds = learner(x_query)
            query_loss = criterion(query_preds, y_query)
            meta_valid_loss += query_loss

        meta_valid_loss = meta_valid_loss / effective_batch_size

        if iter % 10 == 0:
            logger.info(f'Iteration: {iter} Meta Train Loss: {meta_valid_loss.item()}')

        meta_valid_loss.backward()
        opt.step()

    # 验证集（查询集）
    for iter, batch in enumerate(query_dataloader):
        opt.zero_grad()
        effective_batch_size = batch[0].shape[0]
        meta_valid_loss = 0.0
        for i in range(effective_batch_size):
            learner = maml.clone()

            # divide the data into support and query sets
            train_inputs, train_targets = batch[0][i].float(), batch[1][i].float()
            # x_support, y_support = train_inputs[::2], train_targets[::2]
            # x_query, y_query = train_inputs[1::2], train_targets[1::2]
            x_support, y_support = train_inputs[::2], train_targets
            x_query, y_query = train_inputs[1::2], train_targets

            for _ in range(adaptation_steps):  # adaptation_steps
                support_preds = learner(x_support)
                support_loss = criterion(support_preds, y_support)
                learner.adapt(support_loss)

            query_preds = learner(x_query)
            query_loss = criterion(query_preds, y_query)
            meta_valid_loss += query_loss

        meta_valid_loss = meta_valid_loss / effective_batch_size

        if iter % 10 == 0:
            logger.info(f'Iteration: {iter} Meta Valid Loss: {meta_valid_loss.item()}')

    # for iteration in range(num_iterations):
    #     opt.zero_grad()
    #     meta_train_error = 0.0
    #     meta_train_accuracy = 0.0
    #     meta_valid_error = 0.0
    #     meta_valid_accuracy = 0.0
    #     for task in range(meta_batch_size):
    #         # Compute meta-training loss
    #         learner = maml.clone()
    #         batch = tasksets.train.sample()
    #         evaluation_error, evaluation_accuracy = fast_adapt(batch,
    #                                                            learner,
    #                                                            criterion,
    #                                                            adaptation_steps,
    #                                                            shots,
    #                                                            ways,
    #                                                            device)
    #         evaluation_error.backward()
    #         meta_train_error += evaluation_error.item()
    #         meta_train_accuracy += evaluation_accuracy.item()
    #
    #         # Compute meta-validation loss
    #         learner = maml.clone()
    #         batch = tasksets.validation.sample()
    #         evaluation_error, evaluation_accuracy = fast_adapt(batch,
    #                                                            learner,
    #                                                            criterion,
    #                                                            adaptation_steps,
    #                                                            shots,
    #                                                            ways,
    #                                                            device)
    #         meta_valid_error += evaluation_error.item()
    #         meta_valid_accuracy += evaluation_accuracy.item()
    #
    #     # Print some metrics
    #     print('\n')
    #     print('Iteration', iteration)
    #     print('Meta Train Error', meta_train_error / meta_batch_size)
    #     print('Meta Train Accuracy', meta_train_accuracy / meta_batch_size)
    #     print('Meta Valid Error', meta_valid_error / meta_batch_size)
    #     print('Meta Valid Accuracy', meta_valid_accuracy / meta_batch_size)
    #
    #     # Average the accumulated gradients and optimize
    #     for p in maml.parameters():
    #         p.grad.data.mul_(1.0 / meta_batch_size)
    #     opt.step()
    #
    # meta_test_error = 0.0
    # meta_test_accuracy = 0.0
    # for task in range(meta_batch_size):
    #     # Compute meta-testing loss
    #     learner = maml.clone()
    #     batch = tasksets.test.sample()
    #     evaluation_error, evaluation_accuracy = fast_adapt(batch,
    #                                                        learner,
    #                                                        criterion,
    #                                                        adaptation_steps,
    #                                                        shots,
    #                                                        ways,
    #                                                        device)
    #     meta_test_error += evaluation_error.item()
    #     meta_test_accuracy += evaluation_accuracy.item()
    # print('Meta Test Error', meta_test_error / meta_batch_size)
    # print('Meta Test Accuracy', meta_test_accuracy / meta_batch_size)


if __name__ == '__main__':
    main(ways=5,
         shots=1,
         model_lr=0.001,
         maml_lr=0.005,
         meta_batch_size=16,
         adaptation_steps=10,
         num_iterations=600,
         seed=42)
