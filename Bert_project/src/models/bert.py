import torch
import torch.nn as nn
import os
from transformers import BertModel, BertTokenizer, BertConfig


class Config(object):
    def __init__(self):
        """
        定义模型参数
        配置参数，包含模型和训练所需的各种参数
        """
        self.model_name = "bert"  # 模型名称
        self.data_path = (
            "/Users/chen/heimaPythonSpace/Bert_project/data/data1/"  # 数据集路径
        )
        self.train_path = self.data_path + "train.txt"  # 训练集
        self.dev_path = self.data_path + "dev.txt"  # 验证集
        self.test_path = self.data_path + "test.txt"  # 验证集
        self.class_list = [
            x.strip() for x in open(self.data_path + "class.txt").readlines() if x.strip()
        ]  # 类别列表（过滤空行）
        self.save_path = "/Users/chen/heimaPythonSpace/Bert_project/src/saved_dic"
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.save_path += "/" + self.model_name + ".pt"  # 模型训练结果保存路径
        self.save_path2 = "/Users/chen/heimaPythonSpace/Bert_project/src/saved_dic1"
        if not os.path.exists(self.save_path2):
            os.mkdir(self.save_path2)
        self.save_path2 += "/" + self.model_name + "_quantized.pt"  # 量化模型存储结果
        # Apple GPU (MPS) 加速，不可用时回退到 CPU
        # if torch.backends.mps.is_available():
        #     self.device = torch.device("mps")
        # else:
        self.device = torch.device("cpu")

        self.num_classes = len(self.class_list)  # 类别数量
        self.leanrning_rate = 5e-5  # 学习率
        self.batch_size = 128  # 批次大小
        self.pad_size = 32  # 句子长度
        self.epoches = 1  # 训练轮数

        self.bert_path = '/Users/chen/heimaPythonSpace/Bert_project/data/bert_pretrain/'# bert预训练模型路径
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)#创建分词器
        self.bert_config = BertConfig.from_pretrained(self.bert_path+'bert_config.json')#获取bert的配置参数
        self.hidden_size = self.bert_config.hidden_size

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        #预训练bert模型
        self.bert=BertModel.from_pretrained(config.bert_path,config=config.bert_config)
        #创建线性层映射到类别数
        self.fc=nn.Linear(config.hidden_size,config.num_classes)

    def forward(self, x):
        context=x[0]#输入的句子
        mask=x[2]#对padding部分进行mask
        _, pooled=self.bert(context, attention_mask=mask, return_dict=False)
        out=self.fc(pooled)
        return out

