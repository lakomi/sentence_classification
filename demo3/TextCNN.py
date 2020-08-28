# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：sentence_classification -> TextCNN
@IDE    ：PyCharm
@Author ：qs
@Date   ：2020/8/28 16:26
@Desc   ：
=================================================='''
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, num_filters, kernel_sizes, vocab_size, embedding_size, num_classes, sequence_length):
        super(TextCNN, self).__init__()
        self.num_filter_total = num_filters * len(kernel_sizes)
        # 特征向量化
        self.embed = nn.Embedding(vocab_size, embedding_size)
        # 卷积层
        self.convs = nn.ModuleList([nn.Conv2d(1, num_filters, (size, embedding_size)) for size in kernel_sizes])
        # 全连接
        self.fc1 = nn.Linear(self.num_filter_total, num_classes, bias=False)
        self.Bias = nn.Parameter(torch.ones([num_classes]))
        self.sequence_length = sequence_length
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        x = self.embed(x)
        x = x.unsqueeze(1)

        pooled_output = []
        for i, conv in enumerate(self.convs):
            # conv:[input_channel(=1), output_channel(=3), (filter_height, filter_width), bias_option]
            h = F.relu(conv(x))
            # mp:((filter_height, filter_width))
            mp = nn.MaxPool2d((self.sequence_length - self.kernel_sizes[i] + 1, 1))
            # pooled : [batch_size(=6), output_height(=1), output_width(=1), output_channel(=3)]
            pooled = mp(h).permute(0, 3, 2, 1)
            pooled_output.append(pooled)

        # [batch_size(=6), output_height(=1), output_width(=1), output_channel(=3) * 3]
        h_pool = torch.cat(pooled_output, len(self.kernel_sizes))
        # [batch_size(=6), output_height * output_width * (output_channel * 3)]
        h_pool_flat = torch.reshape(h_pool, [-1, self.num_filter_total])
        # [batch_size, num_classes]。将上述的结果全连接。
        model = self.fc1(h_pool_flat) + self.Bias
        return model
