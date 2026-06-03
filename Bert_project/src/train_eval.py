import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
import torch.optim as optim
from tqdm import tqdm
import math
import logging


def loss_fn(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)


def train(config, model, train_iter, dev_iter):
    """
    模型训练函数
    参数：
    config:模型配置参数
    model:模型
    train_iter:训练数据迭代器
    dev_iter:验证数据迭代器
    """
    # 记录开始时间
    start_time = time.time()
    # 参数优化器设置
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        # 需要权重衰减的参数
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        # 不需要权重衰减的参数
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    # 设置优化器
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=config.leanrning_rate)
    dev_best_loss = float("inf")
    # 将模型设置为训练模式
    model.train()
    # 遍历每个轮次epoch
    for epoch in range(config.epoches):
        total_batch = 0
        print("Epoch [{}/{}]".format(epoch + 1, config.epoches))
        # 遍历每个批次batch
        for i, (trains, labels) in enumerate(tqdm(train_iter)):
            # 模型前向传播
            output = model(trains)
            # 梯度清零
            model.zero_grad()
            # 计算损失
            loss = loss_fn(output, labels)
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()
            # 每100个批次查看一次准确率
            if total_batch % 100 == 0 and total_batch != 0:
                true = labels.detach().cpu()
                pred = torch.max(output.detach(), 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, pred)
                # 评估验证集效果
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = "*"
                else:
                    improve = ""
                time_dif = get_time_dif(start_time)
                # 输出训练集和验证集上的效果
                msg = "Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}"
                print(
                    msg.format(
                        total_batch,
                        loss.item(),
                        train_acc,
                        dev_loss,
                        dev_acc,
                        time_dif,
                        improve,
                    )
                )
                model.train()
            total_batch += 1


def evaluate(config, model, data_iter, test=False):
    """
    评估函数
    参数：
    - config: 配置信息对象
    - model: 待评估的模型对象
    - data_iter: 数据迭代器对象
    - test: 是否为测试集,默认为False
    """
    # 模型转为评估模式
    model.eval()

    loss_total = 0.0
    #预测结果
    predict_all = np.array([], dtype=int)
    #label信息
    labels_all = np.array([], dtype=int)
    #不进行梯度计算
    with torch.no_grad():
        #遍历数据集
        for texts, labels in data_iter:
            #将数据送入模型
            outputs = model(texts)
            #损失函数
            loss = F.cross_entropy(outputs, labels)
            #累加损失
            loss_total += loss
            #获取label信息
            labels=labels.data.cpu().numpy()
            #获取预测结果
            preds=torch.max(outputs.data,1)[1].cpu().numpy()
            labels_all=np.append(labels_all,labels)
            predict_all=np.append(predict_all,preds)
    #计算准确率
    acc = metrics.accuracy_score(labels_all,predict_all)

    if test:
        #如果是测试集评估，计算分类报告和混淆矩阵
        report = metrics.classification_report(labels_all,predict_all,target_names=config.class_list,digits=4)#分类报告
        confusion = metrics.confusion_matrix(labels_all,predict_all)#混淆矩阵
        return acc,loss_total/len(data_iter),report,confusion
    else:
        #如果是验证集评估，仅计算准确率和平均损失
        return acc,loss_total/len(data_iter)
    
#测试函数
def test(config,model,test_iter):
    """
    模型测试函数，用于在测试集上评估模型性能
    参数：
    config: 模型配置参数
    model: 训练好的模型
    test_iter: 测试集的数据迭代器
    """
    
    model.eval()

    start_time = time.time()
    #调用evaluate函数   
    test_acc,test_loss,test_report,test_confusion = evaluate(config,model,test_iter,test=True)

    #打印测试信息
    msg='Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss,test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif=get_time_dif(start_time)
    print("Time usage:",time_dif)
