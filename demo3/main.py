# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：sentence_classification -> main
@IDE    ：PyCharm
@Author ：qs
@Date   ：2020/8/28 16:44
@Desc   ：
=================================================='''
from demo3 import train

embedding_size = 2
sequence_length = 3
num_classes = 2
kernel_sizes = [2, 2, 2]
num_filters = 3

# 句子长度为3。手动造训练数据
sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
labels = [1, 1, 1, 0, 0, 0]  # 1 is good, 0 is not good.

word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}
vocab_size = len(word_dict)

model = train.train(word_dict, sentences, labels, num_filters, kernel_sizes, vocab_size, embedding_size, num_classes,
                    sequence_length)

test_text = "sorry hate you"
train.test(test_text, word_dict, model)
