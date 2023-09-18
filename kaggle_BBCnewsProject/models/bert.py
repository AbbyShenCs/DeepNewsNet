# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/data/BBC News Train.csv'                                # 训练集
        self.dev_path = dataset + '/data/BBC News Dev.csv'                                    # 验证集
        self.test_path = dataset + '/data/BBC News Test.csv'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 存储模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 300                                 # 若超过100batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 1                                           # mini-batch大小
        self.pad_size = 512                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-6                                       # 学习率
        #预训练模型相关文件(模型文件.bin、配置文件.json、词表文件vocab.txt)存储路径
        self.bert_path = './bert_pretrain'
        #序列切分工具
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        #隐藏单元数
        self.hidden_size = 768


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        #加载bert预训练模型
        self.bert = BertModel.from_pretrained(config.bert_path)
        #微调
        for param in self.bert.parameters():
            param.requires_grad = True         #finetuning
        #输出层
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)#pooled (batch,hidden_size) cls对应的最后一层的编码向量
        out = self.fc(pooled)#(batch,classes)
        return out