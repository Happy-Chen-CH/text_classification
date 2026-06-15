import numpy as np
import torch
from train_eval import train_kd, train
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, build_dataset_CNN

# 解析命令行参数
parser = argparse.ArgumentParser(description="Chinese Text Classification")
parser.add_argument("--task", type=str, default='train_kd', help="choose a task: trainbert, or train_kd")
args = parser.parse_args()

if __name__ == "__main__":
    # 根据任务类型选择不同的模型和配置
    if args.task == "trainbert":
        model_name = "bert"
        x = import_module("models." + model_name)  # 动态导入模型
        config = x.Config()  # 使用模型的配置

        # 初始化
        np.random.seed(1)
        torch.manual_seed(1)
        if config.device.type != 'mps':
            torch.cuda.manual_seed_all(1)
            torch.backends.cudnn.deterministic = True  # 保证每次结果一样
        
        # 数据集构建
        print("Loading data for Bert Model...")
        train_data, dev_data, test_data = build_dataset(config)  # 构建数据集
        train_iter = build_iterator(train_data, config)  # 构建数据迭代器
        dev_iter = build_iterator(dev_data, config)
        test_iter = build_iterator(test_data, config)
        
        # 模型实例化与训练
        model = x.Model(config).to(config.device)  # 实例化模型，并将模型移动到设备上
        
        train(config, model, train_iter, dev_iter, test_iter)

    if args.task == "train_kd":
        # 加载bert模型
        model_name = "bert"
        bert_module = import_module("models." + model_name)
        bert_config = bert_module.Config()  # 使用BERT模型的配置

        # 加载cnn模型
        model_name = "textCNN"
        cnn_module = import_module("models." + model_name)
        cnn_config = cnn_module.Config()  # 使用TextCNN模型的配置

        # 初始化
        np.random.seed(1)
        torch.manual_seed(1)
        if bert_config.device.type != 'mps':
            torch.cuda.manual_seed_all(1)
            torch.backends.cudnn.deterministic = True  # 保证每次结果一样
        
        # 构建bert数据集，因为只需要训练结果作为软目标，这里不需要dev_iter和test_iter
        bert_train_data, _, _ = build_dataset(bert_config)
        bert_train_iter = build_iterator(bert_train_data, bert_config)
        
        # 构建cnn数据集
        vocab, cnn_train_data, cnn_dev_data, cnn_test_data = build_dataset_CNN(cnn_config)
        cnn_train_iter = build_iterator(cnn_train_data, cnn_config)
        cnn_dev_iter = build_iterator(cnn_dev_data, cnn_config)
        cnn_test_iter = build_iterator(cnn_test_data, cnn_config)
        cnn_config.n_vocab = len(vocab)
        
        # 加载训练好的teacher模型
        bert_model = bert_module.Model(bert_config).to(bert_config.device)
        
        # 加载student模型
        cnn_model = cnn_module.Model(cnn_config).to(cnn_config.device)
        
        print("Teacher and student models loaded, start training")
        
        train_kd(cnn_config, bert_model, cnn_model,
                 bert_train_iter, cnn_train_iter, cnn_dev_iter, cnn_test_iter)