import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class Config(object):
    def __init__(self):
        self.model_name = "textCNN"
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.data_path = os.path.join(base_dir, "data", "data") + "/"
        self.train_path = self.data_path + "train.txt"  # 训练集
        self.dev_path = self.data_path + "dev.txt"  # 验证集
        self.test_path = self.data_path + "test.txt"  # 测试集
        self.class_list = [x.strip() for x in open(self.data_path+"class.txt", encoding="utf-8").readlines()]
        self.vocab_path = self.data_path + "vocab.pkl"  # 词表
        self.save_path = os.path.join(base_dir, "src", "saved_dic")
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.save_path += "/" + self.model_name + ".pt"  # 模型训练结果
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设备

        self.dropout = 0.5 #随机失活
        self.require_improvement = 1000
        self.num_classes = len(self.class_list)
        self.num_epochs = 3
        self.n_vocab = 0
        self.batch_size = 128
        self.pad_size = 32
        self.learning_rate = 1e-3
        self.embed = 300
        self.filter_sizes = (2, 3, 4)
        self.num_filters = 256

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1) # 词嵌入层
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes]
        )   # 卷积层列表，包含不同卷积核大小的卷积层
        self.dropout = nn.Dropout(config.dropout)  # 随机失活层
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)   # 全连接层

    def conv_and_pool(self, x, conv):
        # 卷积和池化操作
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        # 前向传播
        out = self.embedding(x[0])
        out = out.unsqueeze(1)
        # 对每个卷积层进行卷积和池化操作，然后拼接在一起
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)  # 随机失活
        out = self.fc(out)   # 全连接层
        return out