import torch
import torch.nn as nn
import os
from transformers import BertModel, BertTokenizer, BertConfig

class Config(object):
    def __init__(self):
        """
        配置类,包含模型和训练所需各种参数
        """
        self.model_name = 'bert'
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.data_path = os.path.join(base_dir, 'data', 'data') + '/'
        #训练集
        self.train_path = self.data_path + 'train.txt'
        #验证集
        self.dev_path = self.data_path + 'dev.txt'
        #测试集
        self.test_path = self.data_path + 'test.txt'
        self.class_list = [x.strip() for x in open(
            self.data_path + 'class.txt').readlines()]#类别名单
        project_root = os.path.dirname(base_dir)
        self.save_path = os.path.join(project_root, 'Bert_project', 'src', 'saved_dic')
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.save_path+='/'+self.model_name+'.pt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = len(self.class_list)#类别数
        self.num_epochs = 1
        self.batch_size = 128
        self.pad_size = 32
        self.learning_rate = 5e-5
        self.bert_path = 'bert-base-chinese'  # 从HuggingFace Hub在线拉取bert预训练模型
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.bert_config = BertConfig.from_pretrained(self.bert_path)
        self.hidden_size = self.bert_config.hidden_size

class Model(nn.Module):
    def __init__(self, config):
        """
        模型类,初始参数,及线性层设置
        """
        super(Model,self).__init__()
        #预训练bert模型
        self.bert=BertModel.from_pretrained(config.bert_path,config=config.bert_config)
        #全连接层，用于文本分类
        self.fc=nn.Linear(config.hidden_size,config.num_classes)

    def forward(self,x):
        """
        前向传播

        参数：
            x:输入的文本,包含句子,句子长度,填充掩码
        """
        context=x[0]
        mask=x[2]
        _,pooled = self.bert(context,attention_mask=mask,return_dict=False)
        #模型输出,用于文本分类
        out = self.fc(pooled)
        return out


