import time

import torch
import torch.nn as nn
import torch.optim as optim
import global_config as cfg
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, dataset, random_split
from MetaRegressionModel import RegressionModel
import matplotlib.pyplot as plt
from FeatureExtraction import FeatureExtraction
from DataPreprocess import get_single_column_data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

logger = cfg.logger
# os.environ['NUMEXPR_MAX_THREADS'] = r'16'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def main():
    # 准备数据集
    organ = 'blood'
    if organ == 'brain':
        train_data = "D:\\ML\\Medical Data Process\\DL\\Datasets\\brain_322_dataset.pt"
        train_tensor_data = torch.load(train_data, map_location=device)

        eval_data = "D:\\ML\\Medical Data Process\\DL\\ExtenalDatasets\\brain_9_dataset.pt"
        eval_tensor_data = torch.load(eval_data, map_location=device)
    if organ == 'blood':
        train_data = "D:\\ML\\Medical Data Process\\DL\\Datasets\\blood_450_dataset.pt"
        train_tensor_data = torch.load(train_data, map_location=device)

        eval_data = "D:\\ML\\Medical Data Process\\DL\\ExtenalDatasets\\blood_12_dataset.pt"
        eval_tensor_data = torch.load(eval_data, map_location=device)

    # 定义超参数
    input_size = 50
    hidden_size = 256
    output_size = 1
    num_epochs = 600
    learning_rate = 0.001
    l2 = 1e-5

    logger.info(f"Input size: {input_size}")
    logger.info(f"Hidden size: {hidden_size}")
    logger.info(f"Output size: {output_size}")
    # logger.info(f"Number of epoch: {num_epochs}")
    logger.info(f"Learning rate: {learning_rate}")
    # logger.info(f"L2 normalization: {l2}")

    # 包装数据集
    # x_train, y_train = torch.Tensor(x_train.values).to(device), \
    #     torch.Tensor(y_train.values).resize_(y_train.shape[0], 1).to(device)
    # x_test, y_test = torch.Tensor(x_test.values).to(device), \
    #     torch.Tensor(y_test.values).resize_(y_test.shape[0], 1).to(device)
    # train_data = dataset.TensorDataset(x_train, y_train)
    # test_data = dataset.TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_tensor_data, shuffle=True)
    test_loader = DataLoader(eval_tensor_data, shuffle=True)

    model = RegressionModel(input_size, hidden_size, output_size).to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loss = []
    test_loss = []
    start_time = time.time()
    logger.info("Training...")
    # 训练模型
    i = 1
    for inputs, targets in train_loader:
        # 前向传播
        inputs = inputs[:, :inputs.shape[1]//2]
        inputs = inputs.type(torch.FloatTensor).to(device)
        targets = targets.type(torch.FloatTensor).to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        if i % 10 == 0:
            logger.info(f"Batch {i} loss: {loss.item()}")
        i = i + 1
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    end_time = time.time()
    logger.info("Training spent: " + str(round(end_time - start_time, 1)) + "s")

    logger.info("Evaluating...")
    # 预测模型
    i = 1
    for inputs, targets in test_loader:
        inputs = inputs.type(torch.FloatTensor).to(device)
        targets = targets.type(torch.FloatTensor).to(device)
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            logger.info(f"Batch {i} loss: {loss.item()}")
            i = i + 1
            logger.info(f"Pred: {round(outputs.item(), 3)}, ground true: {round(targets.item(), 3)}\n")
        # 反向传播和优化
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

    # for epoch in range(num_epochs):
    #
    #
    #     if epoch % 10 == 0:
    #         model.eval()
    #         with torch.no_grad():
    #             train_epoch_loss = sum(criterion(model(xb), yb) for xb, yb in train_loader)
    #             test_epoch_loss = sum(criterion(model(xb), yb) for xb, yb in test_loader)
    #         train_loss.append(train_epoch_loss.rawdata.item() / len(train_loader))
    #         test_loss.append(test_epoch_loss.rawdata.item() / len(test_loader))
    #         template = "epoch:{:2d}, 训练损失:{:.5f}, 验证损失:{:.5f}"
    #         logger.info(template.format(epoch, np.sqrt(train_epoch_loss.rawdata.item() / len(train_loader)),
    #                                     np.sqrt(test_epoch_loss.rawdata.item() / len(test_loader))))


if __name__ == '__main__':
    main()

