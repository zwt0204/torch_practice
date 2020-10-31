#!/user/bin/env python
# coding=utf-8
"""
@file: lstm.py
@author: zwt
@time: 2020/10/30 15:26
@desc:https://github.com/renjunxiang/Poetry_Generate_PyTorch/tree/master/data
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pickle
from torch import optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# train
EPOCH = 50
BATCH_SIZE = 64
LR = 0.001  # 学习率
LOG_BATCH_NUM = 50  # 日志打印频率
VOCAB_SIZE = 6630

# test
# MODEL_PATH_RNN = '../model/RNN.pth'  # rnn模型位置


class RNN(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, bidirectional=True)
        self.output = nn.Linear(hidden_size*2, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, (_, _) = self.rnn(x)
        x = self.output(out)
        x = x.view(-1, x.size()[-1])
        return x


# 定义数据读取方式
class DatasetRNN(Dataset):
    def __init__(self, x_seq, y_seq):
        self.x_seq = x_seq
        self.y_seq = y_seq

    def __getitem__(self, index):
        return self.x_seq[index], self.y_seq[index]

    def __len__(self):
        return len(self.x_seq)


def train():
    with open('../data/x_seq.pkl', 'rb') as f:
        x_seq = pickle.load(f)
    with open('../data/y_seq.pkl', 'rb') as f:
        y_seq = pickle.load(f)

    # x_seq是[[1, 5, ....], [0, 5, ...], [22, 3, ...]]
    # 定义训练批处理数据
    trainloader = torch.utils.data.DataLoader(
        dataset=DatasetRNN(x_seq[:-1000], y_seq[:-1000]),
        batch_size=BATCH_SIZE, shuffle=True)

    testloader = torch.utils.data.DataLoader(
        dataset=DatasetRNN(x_seq[-1000:], y_seq[-1000:]),
        batch_size=BATCH_SIZE, shuffle=True)

    # 定义损失函数loss function和优化方式
    net = RNN(VOCAB_SIZE, 300, 300, 1).to(device)
    # net = NET_RNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=0.0005)

    for epoch in range(EPOCH):
        sum_loss = 0.0
        # 数据读取
        for i, data in enumerate(trainloader):
            x_seq_batch, y_seq_batch = data
            x_seq_batch, y_seq_batch = x_seq_batch.to(device), y_seq_batch.to(device)
            y_seq_batch = y_seq_batch.flatten()

            # 梯度清零
            optimizer.zero_grad()

            outputs = net(x_seq_batch)
            loss = criterion(outputs, y_seq_batch)
            loss.backward()
            optimizer.step()

            # 每训练LOG_BATCH_NUM个batch打印一次平均loss
            sum_loss += loss.item()
            if i % LOG_BATCH_NUM == LOG_BATCH_NUM - 1:
                print('[%d, %d] loss: %.03f'
                      % (epoch + 1, i + 1, sum_loss / LOG_BATCH_NUM))
                sum_loss = 0.0

        # 每跑完一次epoch测试一下准确率
        with torch.no_grad():
            correct = 0
            total = 0
            for data in testloader:
                x_seq_batch, y_seq_batch = data
                x_seq_batch, y_seq_batch = x_seq_batch.to(device), y_seq_batch.to(device)
                y_seq_batch = y_seq_batch.flatten()
                outputs = net(x_seq_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += y_seq_batch.size(0)
                correct += (predicted == y_seq_batch).sum()
            # print('第%d个epoch的识别准确率为：%d%%' % (epoch + 1, torch.floor_divide(correct, total)))
            print('第%d个epoch的识别准确率为：%f' % (epoch + 1, correct.__float__() / total))
        torch.save(net.state_dict(), '%s/%03d.pth' % ('../model', epoch + 1))


if __name__ == '__main__':
    train()



