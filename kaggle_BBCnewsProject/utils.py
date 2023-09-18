# coding: UTF-8
import torch
from tqdm import tqdm
import time
from datetime import timedelta
import pandas as pd
PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号
label_dict={'business':0,'entertainment':1,'politics':2,'sport':3,'tech':4}

def build_dataset(config):

    def load_dataset(path, pad_size=32):
        contents = []
        train_df = pd.read_csv(path, sep=',')
        train_df.dropna(inplace=True)
        label_flag=0
        for index , row in train_df.iterrows():
            content=row["Text"]
            if(path != 'BBCNews' + '/data/BBC News Test.csv'):
                label=row['Category']
                label_flag=1
            #print(label)
            token = config.tokenizer.tokenize(content)#分字(汉语 character-level) bert内置的tokenizer
            if(label_flag):
                    label_ids = label_dict[label]
            token = [CLS] + token#头部加入 [CLS] token
            seq_len = len(token)#文本实际长度（填充或截断之前）
            mask = []#区分填充部分和非填充部分
            token_ids = config.tokenizer.convert_tokens_to_ids(token)#把tokenizer转换为索引（基于下载的词表文件）
            #print(label_ids[0])

            if pad_size:#长截短填
                if len(token) < pad_size:#序列长度小于 填充长度
                    mask = [1] * len(token_ids) + [0] * (pad_size - len(token))#mask 填充部分对应0 非填充部分为1
                    token_ids += ([0] * (pad_size - len(token)))#用0作填充
                else:#此时没有填充 序列长度大于填充长度
                    mask = [1] * pad_size#全部都是非填充
                    token_ids = token_ids[:pad_size]#截断
                    seq_len = pad_size #实际长度为填充长度
            if(label_flag):
                    contents.append((token_ids, int(label_ids), seq_len, mask))#[([...],label,seq_len,[...])]
            else:
                    contents.append((token_ids, seq_len, mask))#[([...],label,seq_len,[...])]
        return contents
        
    # 分别对训练集、验证集、测试集进行处理
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    # 返回预处理好的训练集、验证集、测试集
    return train, dev, test


class DatasetIterater(object):#自定义数据集迭代器 ,变成batch形式
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size #得到batch数量
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0: #不能整除
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        # 转换为tensor 并 to(device)
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)  #输入序列
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)  #标签

# seq_len为文本的实际长度（不包含填充的长度） 转换为tensor 并 to(device)   # pad前的长度(超过pad_size的设为pad_size)
      
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:#当数据集大小 不整除 batch_size时，构建最后一个batch
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)#把最后一个batch转换为tensor 并 to(device
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:#构建每一个batch
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)#把当前batch转换为tensor 并 to(device)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1#不整除 batch数加1
        else:
            return self.n_batches

class TestDatasetIterater(object):#自定义数据集迭代器 ,变成batch形式
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size #得到batch数量
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0: #不能整除
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        # 转换为tensor 并 to(device)
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)  #输入序列

# seq_len为文本的实际长度（不包含填充的长度） 转换为tensor 并 to(device)   # pad前的长度(超过pad_size的设为pad_size)
      
        seq_len = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len, mask)

    def __next__(self):
        if self.residue and self.index == self.n_batches:#当数据集大小 不整除 batch_size时，构建最后一个batch
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)#把最后一个batch转换为tensor 并 to(device
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:#构建每一个batch
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)#把当前batch转换为tensor 并 to(device)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1#不整除 batch数加1
        else:
            return self.n_batches


def build_test_iterator(dataset, config):#构建数据集迭代器
    iter = TestDatasetIterater(dataset, config.batch_size, config.device)
    return iter

def build_iterator(dataset, config):#构建数据集迭代器
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
