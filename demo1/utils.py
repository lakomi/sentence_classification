# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：sentence_classification -> utils
@IDE    ：PyCharm
@Author ：qs
@Date   ：2020/8/15 11:14
@Desc   ：
=================================================='''
import torchtext.datasets as datasets
import torchtext.data as data


# load SST dataset
def sst(text_field, label_field, batch_sizes, **kwargs):
    # SST为torchtext提供的常用文本数据集，可以直接加载使用。
    # splits()可以同时读取训练集，验证集，测试集
    train_data, dev_data, test_data = datasets.SST.splits(text_field, label_field, fine_grained=True)
    # build_vocab()构建词表
    text_field.build_vocab(train_data, dev_data, test_data)
    label_field.build_vocab(train_data, dev_data, test_data)

    # iterator是torchtext到模型的输出。提供了对数据的一般处理方式（打乱、排序等）。
    # splits(（加载的数据集）， Batch 大小，**kwargs)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits((train_data, dev_data, test_data), batch_sizes=(
        batch_sizes, len(dev_data), len(test_data)), **kwargs)

    return train_iter, dev_iter, test_iter
