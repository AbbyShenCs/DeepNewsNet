# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse

# 声明argparse对象 可附加说明
parser = argparse.ArgumentParser(description='Chinese Text Classification')

# 添加参数
# 模型是必须设置的参数(required=True) 类型是字符串
parser.add_argument('--model', type=str, required=True,
                    help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
# embedding随机初始化或使用预训练词或字向量 默认使用预训练
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
# 基于词还是基于字 默认基于字
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')

# 解析参数
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集

    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'  # 默认使用搜狗预训练字向量
    if args.embedding == 'random':  # 如果embedding参数设置为random
        embedding = 'random'

    # 获取选择的模型名字
    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer

    # 导入数据预处理与加载函数
    if model_name == 'FastText':  # 如果所选模型名字为FastText 由于增加了bi-gram tri-gram特征 会有不同的行为
        from utils_fasttext import build_dataset, build_iterator, get_time_dif

        embedding = 'random'  # 此时embedding需要设置为随机初始化
    else:  # 其他模型统一处理
        from utils import build_dataset, build_iterator, get_time_dif

    x = import_module('models.' + model_name)  # 根据所选模型名字在models包下 获取相应模块(.py)
    config = x.Config(dataset, embedding)  # 每一个模块(.py)中都有一个模型定义类 和与该模型相关的配置类(定义该模型的超参数) 初始化配置类的对象

    # 设置随机种子 确保每次运行的条件(模型参数初始化、数据集的切分或打乱等)是一样的
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    # 保证每次结果一样 指定了随机种子之后，下一次无论执行多少次，每次结果一样
    # 为了之后在复现的过程中调节超参数会对结果产生影响，如果随机策略改变，参数改变，无法验证调节参数是否有效，因此要保证随机种子一致

    start_time = time.time()
    print("Loading data...")
    # 数据预处理
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)  # 构建词典、训练集、验证集、测试集
    # 构建训练集、验证集、测试集迭代器/生成器（节约内存、避免溢出）
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 构造模型对象
    config.n_vocab = len(vocab)  # 词典大小可能不确定，在运行时赋值
    model = x.Model(config).to(config.device)  # 构建模型对象 并to_device

    if model_name != 'Transformer':  # 如果不是Transformer模型 则使用自定义的参数初始化方式
        init_network(model)  # 也可以采用之前达观杯中的做法 把自定义模型参数的函数 放在模型的定义类中 在__init__中执行
    print(model.parameters)

    # 训练、验证和测试
    train(config, model, train_iter, dev_iter, test_iter)