# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：sentence_classification -> model
@IDE    ：PyCharm
@Author ：qs
@Date   ：2020/8/15 18:30
@Desc   ：
=================================================='''
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_Text(nn.Module):
    def __init__(self, args):
        super(CNN_Text, self).__init__()
        self.args = args

        # 词典的大小尺寸
        self.embed_num = args.embed_num
        # 嵌入向量的维度，即用多少维来表示一个符号
        self.embed_dim = args.embed_dim
        # 最后类别数
        self.class_num = args.class_num
        # 核的数量
        self.kernel_num = args.kernel_num
        # 卷积核大小
        self.kernel_sizes = args.kernel_sizes

        # nn.Embedding,是一个简单、存储固定大小的词典 的嵌入向量的查找表。即给一个编号，嵌入层就能返回这个编号对应的嵌入向量。
        # Embeding层 将现实客观特征转成电脑识别的特征，即特征向量化。  定义词嵌入。V个单词，维度D
        self.embed = nn.Embedding(self.embed_num, self.embed_dim)
        # 建立多个连续的卷积层
        self.convs = nn.ModuleList([nn.Conv2d(1, self.kernel_num, (K, self.embed_dim)) for K in self.kernel_sizes])
        # 全连接层，最后的输出通道数为类别数
        self.fc1 = nn.Linear(len(self.kernel_sizes) * self.kernel_num, self.class_num)

        # 丢弃比率
        self.dropout = nn.Dropout(args.dropout)

        if self.args.static:
            self.embed.weight.requires_grad = False

    # 此函数是 数据怎么在刚搭建的网络中流动的写出来
    def forward(self, x):
        # 以下将数据在网络层中怎么走的说清楚
        x = self.embed(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]

        x = torch.cat(x, 1)

        x = self.dropout(x)

        logit = self.fc1(x)
        return logit
