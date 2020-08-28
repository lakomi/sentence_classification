# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：sentence_classification -> train
@IDE    ：PyCharm
@Author ：qs
@Date   ：2020/8/28 19:22
@Desc   ：
=================================================='''
from demo3 import TextCNN
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# 训练
def train(word_dict, sentences, labels, num_filters, kernel_sizes, vocab_size, embedding_size, num_classes,
          sequence_length):
    # 创建模型
    model = TextCNN.TextCNN(num_filters, kernel_sizes, vocab_size, embedding_size, num_classes, sequence_length)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    inputs = torch.LongTensor([np.asarray([word_dict[n] for n in sen.split()]) for sen in sentences])
    target = torch.LongTensor([out for out in labels])

    for epoch in range(5000):
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, target)

        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'lost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()
    return model


# 测试
def test(test_text, word_dict, model):
    tests = [np.asarray([word_dict[n] for n in test_text.split()])]
    test_batch = torch.LongTensor(tests)

    # 通过model获得预测结果
    predict = model(test_batch).data.max(1, keepdim=True)[1]
    if predict[0][0] == 0:
        print(test_text, "is Bad Mean...")
    else:
        print(test_text, "is Good Mean!!")
