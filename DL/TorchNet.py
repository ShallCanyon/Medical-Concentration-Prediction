import time

import torch
import torch.nn as nn
import torch.optim as optim
import global_config as cfg
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, dataset
import matplotlib.pyplot as plt
from FeatureExtraction import FeatureExtraction
from DataPreprocess import get_single_column_data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

logger = cfg.logger
# os.environ['NUMEXPR_MAX_THREADS'] = r'16'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)


def main():
    # 准备数据集
    organ_name = 'brain'
    certain_time = 60
    desc_csv = f"D:\\ML\\Medical Data Process\\{cfg.parent_folder}\\{cfg.filetime}\\{organ_name}-{certain_time}min-desc.csv"
    x, y, _ = get_single_column_data(desc_csv)
    sc = StandardScaler()
    x = FeatureExtraction(X=x, y=y).feature_extraction(RFE=True)
    x = pd.DataFrame(sc.fit_transform(x))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    sample_num, desc_num = x.shape[0], x.shape[1]

    # 定义超参数
    input_size = desc_num
    hidden_size = 256
    output_size = 1
    num_epochs = 600
    learning_rate = 0.001
    l2 = 1e-5

    logger.info(f"Input size: {input_size}")
    logger.info(f"Hidden size: {hidden_size}")
    logger.info(f"Output size: {output_size}")
    logger.info(f"Number of epoch: {num_epochs}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"L2 normalization: {l2}")

    # 包装数据集
    x_train, y_train = torch.Tensor(x_train.values).to(device), \
        torch.Tensor(y_train.values).resize_(y_train.shape[0], 1).to(device)
    x_test, y_test = torch.Tensor(x_test.values).to(device), \
        torch.Tensor(y_test.values).resize_(y_test.shape[0], 1).to(device)
    train_data = dataset.TensorDataset(x_train, y_train)
    test_data = dataset.TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_data, shuffle=True)
    test_loader = DataLoader(test_data, shuffle=True)

    model = RegressionModel(input_size, hidden_size, output_size).to(device)
    # model = nn.Sequential(
    #     nn.Linear(input_size, hidden_size),
    #     nn.ReLU(),
    #     nn.Linear(hidden_size, output_size)
    # )

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2)

    train_loss = []
    test_loss = []
    start_time = time.time()
    # 训练模型
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #
            # pred = model(xb)
            # loss = criterion(pred, yb)
            # loss.requires_grad_(True)
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                train_epoch_loss = sum(criterion(model(xb), yb) for xb, yb in train_loader)
                test_epoch_loss = sum(criterion(model(xb), yb) for xb, yb in test_loader)
            train_loss.append(train_epoch_loss.data.item() / len(train_loader))
            test_loss.append(test_epoch_loss.data.item() / len(test_loader))
            template = "epoch:{:2d}, 训练损失:{:.5f}, 验证损失:{:.5f}"
            logger.info(template.format(epoch, np.sqrt(train_epoch_loss.data.item() / len(train_loader)),
                                        np.sqrt(test_epoch_loss.data.item() / len(test_loader))))
        # inputs, targets = x_train, y_train
        #
        # # 前向传播
        # optimizer.zero_grad()
        # outputs = model(inputs)
        # loss = criterion(outputs, targets)
        #
        # # 反向传播和优化
        # loss.backward()
        # optimizer.step()
        #
        # if (epoch + 1) % 5 == 0:
        #     print('Epoch [%d/%d], Loss: %.4f' % (epoch + 1, num_epochs, loss.item()))
    end_time = time.time()
    print("Training spent: " + str(round(end_time - start_time, 1)) + "s")

# 定义模型
class RegressionModel(nn.Module):
    def __init__(self, input_size, n_hidden, output_size=1):
        super(RegressionModel, self).__init__()
        # self.linear = nn.Linear(input_size, output_size)
        self.hidden1 = nn.Linear(input_size, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        self.predict = nn.Linear(n_hidden, output_size)

    def forward(self, x):
        out = self.hidden1(x)
        out = F.relu(out)
        out = self.hidden2(out)
        out = torch.sigmoid(out)
        out = self.predict(out)
        return out


if __name__ == '__main__':
    main()

