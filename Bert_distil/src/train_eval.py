import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from transformers.optimization import AdamW
from tqdm import tqdm
import math
import logging

def fetch_teacher_outputs(teacher_model, train_iter):
    # 将教师模型设置为评估（推断）模式，避免在获取输出时进行梯度计算
    teacher_model.eval()
    # 用于存储教师模型对训练集的输出
    teacher_outputs = []
    # 禁用梯度计算
    with torch.no_grad():
        # 遍历训练集数据
        for i, (data_batch, labels_batch) in enumerate(train_iter):
            # 获取教师模型的输出
            outputs = teacher_model(data_batch)
            # 将输出添加到列表中
            teacher_outputs.append(outputs)
    # 返回教师模型对训练集的所有输出
    return teacher_outputs

# 交叉熵损失: 训练bert模型
def loss_fn(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)

# KL散度损失（要求student输入为log-probabilities,软目标为probabilities）
criterion = nn.KLDivLoss()

# 定义蒸馏损失函数
def loss_fn_kd(outputs, labels, teacher_outputs):
    # 设置两个重要超参数
    alpha = 0.8
    T = 2

    # 1.学生网络的带有T参数的log_softmax输出分布
    output_student = F.log_softmax(outputs / T, dim=1)
    # 2.教师网络的带有T参数的softmax输出分布
    output_teacher = F.softmax(teacher_outputs / T, dim=1)

    # 3.计算软目标损失,使用KLDivLoss(),第一个参数为student网络输出, 第二个参数为teacher网络输出
    soft_loss = criterion(output_student, output_teacher)

    # 4.硬目标损失，学生网络的输出概率和真实标签之间的损失, 因为真实标签是one-hot编码, 因此直接使用交叉熵损失即可
    hard_loss = F.cross_entropy(outputs, labels)

    # 5.计算总损失
    # 原始论文中已经证明, 引入T会导致软目标产生的梯度和真实目标产生的梯度相比只有1/(T*T)
    # 因此计算完软目标的loss值后要乘以T^2.
    KD_loss = soft_loss * alpha * T * T + hard_loss * (1.0 - alpha)
    return KD_loss

def train(config, model, train_iter, dev_iter, test_iter):
    """
    参数:
    - config: 包含超参数和设置的配置对象。
    - model: 要训练的神经网络模型。
    - train_iter: 用于训练数据集的迭代器。
    - dev_iter: 用于验证（开发）数据集的迭代器。
    - test_iter: 用于测试数据集的迭代器。
    """
    # 记录训练开始时间
    start_time = time.time()
    # 将模型设置为训练模式
    model.train()
    # 获取模型参数
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    # 分组参数并设置优化的权重衰减
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        }]
    # 使用AdamW优化器，设置学习率
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    # 记录最佳验证损失
    dev_best_loss = float("inf")
    # 遍历每个epoch
    for epoch in range(config.num_epochs):
        total_batch = 0
        print("Epoch [{}/{}]".format(epoch + 1, config.num_epochs))
        # 遍历训练数据集的每个batch
        for i, (trains, labels) in enumerate(tqdm(train_iter)):
            # 梯度清零
            model.zero_grad()
            # 前向传播
            outputs = model(trains)
            # 计算损失
            loss = loss_fn(outputs, labels)
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            total_batch += 1
            # 每400个batch打印一次训练信息
            if total_batch % 400 == 0 and total_batch > 0:
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                # 在验证集上进行评估
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                # 检查当前模型是否是最佳模型
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    # 当模型有提升时保存模型权重
                    torch.save(model.state_dict(), config.save_path)
                    improve = "*"
                else:
                    improve = ""
                time_dif = get_time_dif(start_time)
                msg = "Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}"
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                # 将模型重新设置为训练模式
                model.train()
    # 在测试集上测试最终的模型
    test(config, model, test_iter)

def train_kd(cnn_config, bert_model, cnn_model,
             bert_train_iter, cnn_train_iter, cnn_dev_iter, cnn_test_iter):
    """
    使用知识蒸馏（Knowledge Distillation）的方式训练模型。

    参数:
    - cnn_config: 包含CNN模型超参数和设置的配置对象。
    - bert_model: BERT模型。
    - cnn_model: CNN模型。
    - bert_train_iter: 用于BERT模型训练的迭代器。
    - cnn_train_iter: 用于CNN模型训练的迭代器。
    - cnn_dev_iter: 用于CNN模型验证的迭代器。
    - cnn_test_iter: 用于CNN模型测试的迭代器。
    """
    # 记录训练开始时间
    start_time = time.time()
    # 获取CNN模型参数
    param_optimizer = list(cnn_model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        }]

    # 使用AdamW优化器，设置学习率
    optimizer = AdamW(optimizer_grouped_parameters, lr=cnn_config.learning_rate)
    # 记录最佳验证损失
    dev_best_loss = float("inf")
    # 将CNN模型设置为训练模式
    cnn_model.train()
    # 将BERT模型设置为评估模式
    bert_model.eval()
    # 获取BERT模型的输出作为教师模型的预测结果
    teacher_outputs = fetch_teacher_outputs(bert_model, bert_train_iter)
    # 遍历每个epoch
    for epoch in range(cnn_config.num_epochs):
        total_batch = 0
        print("Epoch [{}/{}]".format(epoch + 1, cnn_config.num_epochs))
        # 遍历CNN模型训练数据集的每个batch
        for i, (trains, labels) in enumerate(tqdm(cnn_train_iter)):
            # 梯度清零
            cnn_model.zero_grad()
            # 前向传播
            outputs = cnn_model(trains)
            # 计算蒸馏损失
            loss = loss_fn_kd(outputs, labels, teacher_outputs[i])
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            total_batch += 1
            # 每400个batch打印一次训练信息
            if total_batch % 400 == 0 and total_batch > 0:
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                # 在CNN验证集上进行评估
                dev_acc, dev_loss = evaluate(cnn_config, cnn_model, cnn_dev_iter)
                # 检查当前CNN模型是否是最佳模型
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(cnn_model.state_dict(), cnn_config.save_path)
                    improve = "*"
                else:
                    improve = ""
                time_dif = get_time_dif(start_time)
                msg = "Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}"
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                # 将CNN模型重新设置为训练模式
                cnn_model.train()
    # 在CNN测试集上测试最终的CNN模型
    test(cnn_config, cnn_model, cnn_test_iter)

def test(config, model, test_iter):
    """
    模型测试函数，用于在测试集上进行最终的模型测试。
    参数：
    - config: 配置信息对象。
    - model: 待测试的模型。
    - test_iter: 测试集的数据迭代器。
    """
    model.load_state_dict(torch.load(config.save_path,map_location=torch.device(config.device)))
    model.eval()
    start_time = time.time()
    # 调用验证函数计算评估指标
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)

    # 打印测试结果信息:输出测试集上的损失、准确率、分类报告和混淆矩阵等信息
    msg = "Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}"
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

def evaluate(config, model, data_iter, test=False):
    """
    模型评估函数。
    参数：
    - config: 配置信息对象。
    - model: 待评估的模型。
    - data_iter: 数据迭代器。
    - test: 是否为测试集评估。
    """
    model.eval()
    loss_total = 0
    # 预测结果
    predict_all = np.array([], dtype=int)
    # label信息
    labels_all = np.array([], dtype=int)
    # 不进行梯度计算
    with torch.no_grad():
        # 遍历数据集
        for texts, labels in data_iter:
            # 将数据送入网络中
            outputs = model(texts)
            # 损失函数
            loss = F.cross_entropy(outputs, labels)
            # 损失和
            loss_total += loss
            # 获取label信息
            labels = labels.data.cpu().numpy()
            # 获取预测结果
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
    # 计算准确率
    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        # 如果是测试集评估，计算分类报告和混淆矩阵
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    else:
        # 如果是验证集评估，仅返回准确率和平均损失
        return acc, loss_total / len(data_iter)