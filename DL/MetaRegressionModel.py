import torch
from torch import nn
import torch.nn.functional as F


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
#
# class RegressionModel(torch.nn.Module):
#     def __init__(self):
#         super(RegressionModel, self).__init__()
#         self.hidden_size = 32
#         self.fc1 = torch.nn.Linear(1, self.hidden_size)
#         self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
#         self.fc3 = torch.nn.Linear(self.hidden_size, 1)
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
#
#
# class MetaRegressionModel(torch.nn.Module):
#     def __init__(self, model=None):
#         super(MetaRegressionModel, self).__init__()
#         if model is None:
#             self.model = RegressionModel()
#         else:
#             self.model = model
#
#     def forward(self, inputs, params=None):
#         if params is None:
#             params = self.model.parameters()
#         x, y = inputs
#         y_hat = self.model(x, params=params)
#         loss = F.mse_loss(y_hat, y)
#         return loss, y_hat
#
#     def adapt(self, train_inputs, train_outputs):
#         self.model.train()
#         self.meta_optim.zero_grad()
#         train_loss, _ = self.forward((train_inputs, train_outputs))
#         train_loss.backward()
#         self.meta_optim.step()
#
#     def meta_update(self, train_inputs, train_outputs, test_inputs, test_outputs):
#         self.model.train()
#         params = l2l.clone_parameters(self.model.parameters())
#         train_loss, _ = self.forward((train_inputs, train_outputs))
#         grads = torch.autograd.grad(train_loss, params)
#         fast_weights = l2l.update_parameters(self.model.parameters(), grads, step_size=self.step_size)
#         test_loss, y_hat = self.forward((test_inputs, test_outputs), params=fast_weights)
#         return test_loss, y_hat
#
#     def meta_test(self, test_inputs):
#         self.model.eval()
#         y_hat = self.model(test_inputs)
#         return y_hat
