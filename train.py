# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：sentence_classification -> train
@IDE    ：PyCharm
@Author ：qs
@Date   ：2020/8/15 18:33
@Desc   ：
=================================================='''

import os, sys, torch
import torch.autograd as autograd
import torch.nn.functional as F


def train(train_iter, dev_iter, model, args):
    print("train.....")
    if args.cuda:
        model.cuda()

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    best_acc = 0
    last_step = 0
    # 以下开始训练
    model.train()
    # args.epochs次训练
    for epoch in range(1, args.epochs + 1):
        # for epoch in range(1):
        # 遍历训练数据。每次导入batch_sizes大小的数据。此变量在命令行输入时指定，没有指定默认64
        for batch in train_iter:
            # feature 特征  n*batch_sizes，target标签 batch_sizes。
            feature, target = batch.text, batch.label

            # batch first, index align
            # t_()矩阵转置,sub_()减法。target值减一
            feature.t_(), target.sub_(1)
            # print("feature 大小{}".format(feature.size()))

            # 若gpu可用，则对数据处理，使其能够在gpu中使用
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            # 清零。避免累积
            optimizer.zero_grad()
            # 模型处理得到的结果
            model_result = model(feature)  # 64*5
            # 损失
            loss = F.cross_entropy(model_result, target)
            # 反向传播
            loss.backward()
            # 优化
            optimizer.step()

            steps += 1
            # 是否该打印日志
            if steps % args.log_interval == 0:
                # torch.max(input, dim)。input是softmax函数输出的一个tensor，dim是max函数索引的维度（0-每列最大值，1-每行最大值）。
                # 函数返回两个tensor，第一个是每行的最大值，第二个是每行最大值的索引。
                # 当前batch中，预测正确的数量
                row_max_indexs = torch.max(model_result, 1)[1]
                # view()函数 调整原tensor的形状。
                corrects = (row_max_indexs.view(target.size()).data == target.data).sum()

                # 正确率
                accuracy = 100.0 * corrects / batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})\n'.format(steps, loss.item(), accuracy.item(),
                                                                               corrects.item(), batch.batch_size))

            # 是否该测试
            if steps % args.test_interval == 0:
                dev_acc = eval(dev_iter, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        print("saving best....")
                        # save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
            # 是否该保存快照
            elif steps % args.save_interval == 0:
                print("saving....")
                # save(model, args.save_dir, 'snapshot', steps)
        # break


def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.t_(), target.sub_(1)
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        model_result = model(feature)
        loss = F.cross_entropy(model_result, target, size_average=False)

        avg_loss += loss.item()
        corrects += (torch.max(model_result, 1)[1].view(target.size()).data == target.data).sum()

    # 1101
    size = len(data_iter.dataset)
    # 平均损失
    avg_loss = avg_loss / size
    accuracy = 100.0 * corrects / size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, accuracy, corrects, size))
    return accuracy


def predict(text, model, text_field, label_field, cuda_flag):
    assert isinstance(text, str)
    model.eval()

    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]

    x = torch.tensor(text)
    x = autograd.Variable(x)
    if cuda_flag:
        x = x.cuda()
    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    return label_field.vocab.itos[predicted.item() + 1]


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    print("save_dir 为 {}，save_path 为 {}".format(save_dir, save_path))
    torch.save(model.state_dict(), save_path)
