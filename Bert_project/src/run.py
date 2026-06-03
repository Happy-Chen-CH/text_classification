import time
import torch
import numpy as np
from train_eval import train, test
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif


# 命令行参数解析
parser = argparse.ArgumentParser(description="Chinese Text Classification")
parser.add_argument("--model", type=str, default="bert", help="choose a model: bert")
args = parser.parse_args()

if __name__ == "__main__":
    if args.model == "bert":
        # 导入对应模型配置和模型定义
        model_name = "bert"
        x = import_module("models." + model_name)
        config = x.Config()

        # 设置随机种子，保证试验的可重复性
        np.random.seed(3)
        torch.manual_seed(3)
        # MPS 不支持 cuda.manual_seed_all 和 cudnn 设置
        # 如果检测到是 mps 设备，跳过 cuda 相关设置
        if config.device.type != 'mps':
            torch.cuda.manual_seed_all(3)
            torch.backends.cudnn.deterministic = True

        # 构建训练、验证、测试集数据和数据迭代器
        train_data, dev_data, test_data = build_dataset(config)
        train_iter = build_iterator(train_data, config)
        dev_iter = build_iterator(dev_data, config)
        test_iter = build_iterator(test_data, config)
        # 创建模型实例
        model = x.Model(config).to(config.device)
        # 训练模型
        train(config, model, train_iter, dev_iter)
        #在测试集上测试模型性能
        test(config, model, test_iter)
