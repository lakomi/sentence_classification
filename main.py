# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：sentence_classification -> main
@IDE    ：PyCharm
@Author ：qs
@Date   ：2020/8/15 9:56
@Desc   ：主函数
=================================================='''

import argparse  # 命令行解析包

import torch
import torchtext.data as data

import model as model
import train as train
import utils as myutils

# 使用命令行测试代码。需要以下
# 创建一个ArgumentParser 对象。包含将命令行解析成python数据类型所需的全部信息
parser = argparse.ArgumentParser(description='CNN text classificer')
# 添加参数。一个命名或者一个选项字符串的列表，type命令行参数应当被转换成的类型，default 默认值，help简单描述
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [0.001]')
parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 1]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
# 每训练多少次，就输出一次日志
parser.add_argument('-log-interval', type=int, default=10,
                    help='how many steps to wait before logging training status [default: 1]')
# 每训练多少次，就测试一次
parser.add_argument('-test-interval', type=int, default=100,
                    help='how many steps to wait before testing [default: 100]')
# 每训练多少次，就保存一次
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
# 快照保存目录
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=1000,
                    help='iteration numbers to stop without performance increasing')
# 当获得最佳性能时是否保存
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')

# data。action当参数在命令行中出现时使用的动作基本类型。
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')

# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5',
                    help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')

# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')

# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')

# 解析参数
args = parser.parse_args()

# load data。 data.Field()定义样本的处理操作,创建Field对象,这个对象包含了我们打算如何预处理文本数据的信息
text_field = data.Field(lower=True)
label_field = data.Field(sequential=False)

# 获取到数据。 长度为8544,1101,2202
train_iter, dev_iter, test_iter = myutils.sst(text_field, label_field, args.batch_size, device=-1)

# update args and print
# 词典大小
args.embed_num = len(text_field.vocab)
# 分类的类别数
args.class_num = len(label_field.vocab) - 1
args.cuda = (not args.no_cuda) and torch.cuda.is_available();
del args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
# args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
# print("\nParameters:")
# for attr, value in sorted(args.__dict__.items()):
#     print("\t{}={}".format(attr.upper(), value))

# ------------------------------------------------------------------------------------------------------
# model
cnn = model.CNN_Text(args)
if args.snapshot is not None:
    print('\nLoading model from {}...'.format(args.snapshot))
    cnn.load_state_dict(torch.load(args.snapshot))
if args.cuda:
    torch.cuda.set_device(args.device)
    cnn = cnn.cuda()

# train or predict
if args.predict is not None:
    print("predicting...")
    label = train.predict(args.predict, cnn, text_field, label_field, args.cuda)
    print('\n[Text]  {}\n[Label] {}\n'.format(args.predict, label))
elif args.test:
    try:
        train.eval(test_iter, cnn, args)
    except Exception as e:
        print("\nSorry. The test dataset doesn't  exist.\n")
else:
    print()
    try:
        train.train(train_iter, dev_iter, cnn, args)
    except KeyboardInterrupt:
        print('\n' + '-' * 89)
        print('Exiting from training early')
