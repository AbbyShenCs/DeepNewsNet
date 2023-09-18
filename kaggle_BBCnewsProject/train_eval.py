# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from pytorch_pretrained.optimization import BertAdam
label_reverse_dict={0:'business',1:'entertainment',2:'politics',3:'sport',4:'tech'}

# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()#训练模式
    param_optimizer = list(model.named_parameters())
     #下列参数 不进行正则化（权重衰减）
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    #优化器
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * config.num_epochs)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')#跟踪验证集最小的loss 或 最大f1-score或准确率
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    model.train()#训练模式
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)#前向传播 获取输出
            model.zero_grad()#清空梯度
            loss = F.cross_entropy(outputs, labels)#计算交叉熵损失（内部包含softmax log等操作） 可以用nn里面的函数 也可以用F中的函数 labels为整数索引 内部会自动转换为one-hot
            loss.backward()#计算梯度
            optimizer.step()#更新参数
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果 #每100个batch 计算一下在验证集上的指标 或者像之前项目中那样 每一个epoch在验证集上计算相关指标
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)#当前batch上训练集的准确率 因为是类别均衡数据集，所以可以直接用准确率作评估指标
                dev_acc, dev_loss = evaluate(config, model, dev_iter)#计算此时模型在验证集上的损失和准确率
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss#更新验证集最小损失
                    torch.save(model.state_dict(), config.save_path)#保存在验证集上损失最小的参数
                    improve = '*'#效果提升标志
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()#回到训练模式
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 如果长期没有提高 就提前终止
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    test(config, model, test_iter)#模型训练结束后 进行测试


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))#加载使验证集损失最小的参数
    model.eval()#测试模式
    start_time = time.time()
    predict_all = np.array([], dtype=str)#存储验证集所有batch的预测结果
    with torch.no_grad():
        for texts in test_iter:
            outputs = model(texts)
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            predic=label_reverse_dict[int(predic)]
            predict_all = np.append(predict_all, predic)
    np.savetxt("BBCNews/data/BBC News Test.csv", predict_all, delimiter=",",header='label',fmt="%s")

    '''
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    '''
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()#测试模式
    loss_total = 0
    predict_all = np.array([], dtype=int)#存储验证集所有batch的预测结果
    labels_all = np.array([], dtype=int)#存储验证集所有batch的真实标签
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)#计算验证集准确率
    '''
    if test:#如果是测试集的话 计算一下分类报告
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)#计算混淆矩阵
        return acc, loss_total / len(data_iter), report, confusion
    '''
    return acc, loss_total / len(data_iter)#返回准确率和每个batch的平均损失
