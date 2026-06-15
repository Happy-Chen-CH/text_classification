import torch
from tqdm import tqdm
import time
from datetime import timedelta
import os
import pickle as pkl

UNK, PAD, CLS = "[UNK]", "[PAD]", "[CLS]"  # padding符号
MAX_VOCAB_SIZE = 20000  # 词表长度限制


def build_vocab(file_path, tokenizer, max_size, min_freq=1):
    """
    构建词汇表的函数

    参数：
    file_path:语料路径
    tokenizer:分词器
    max_size:最大词表长度
    min_freq:最小词频

    返回：
    vocab_dic(dict):一个字典,将词汇映射到索引的词汇表
    """
    vocab_dic = {}  # 用于存储的字典，键为单词，值为出现的次数
    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            line = line.strip()
            if not line:
                continue
            content = line.split("\t")[0]  # 以制表符分割，去第一列文本
            # 使用指定分词器对文本进行分词，并更新词汇表
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
    # 根据词频对词汇表进行排序，词频从高到低排序（在遍历所有行之后）
    vocab_list = sorted(
        [_ for _ in vocab_dic.items() if _[1] >= min_freq],
        key=lambda x: x[1],
        reverse=True,
    )[:max_size]
    # 将选定词汇构造成字典，键为单词，值为索引
    vocab_dic = {
        word_count[0]: idx for idx, word_count in enumerate(vocab_list)
    }
    # 添加特殊符号到词汇表
    vocab_dic.update(
        {UNK: len(vocab_dic), PAD: len(vocab_dic) + 1, CLS: len(vocab_dic) + 2}
    )
    return vocab_dic


def build_dataset_CNN(config):
    """
    构建数据集的函数

    参数：
    config:配置信息

    返回：
    vocab(dict):词汇表
    train(list):训练集
    dev(list):验证集
    test(list):测试集
    """
    # 定义字符级别的分词器
    tokenizer = lambda x: [y for y in x]
    # 检查是否存在词汇表文件size已经存在，存在则直接加载，否则创建新的词汇表
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, "rb"))
    else:
        # 构建词汇表
        vocab = build_vocab(
            config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1
        )
        # 保存词汇表
        pkl.dump(vocab, open(config.vocab_path, "wb"))
    print("Vocab size:", len(vocab))

    # 定义加载数据集的辅助函数,加载训练集等
    def load_dataset(path, pad_size=32):
        """
        加载数据集的辅助函数

        参数：
        path:数据集路径
        pad_size:句子长度

        返回：
        contents(list):一个列表,列表中的元素为元组,元组包含一个句子的id列表和一个标签id
        """
        contents = []
        with open(path, "r", encoding="utf-8") as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split("\t")
                words_line = []
                token = tokenizer(content)
                seq_len = len(token)
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size
                # 将词转化为对应id
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                # 将数据添加到contents列表中
                contents.append((words_line, int(label), seq_len))
        return contents

    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return vocab, train, dev, test


# 数据迭代器类
class DatasetIterater(object):
    def __init__(self, batches, batch_size, device, model_name):
        """
        初始化的函数

        参数：
        batches:数据集
        batch_size:批次大小
        device:设备
        model_name:模型名称

        返回：
        None
        """
        self.index = 0  # 当前批次的索引
        self.batch_size = batch_size  # 批次大小
        self.batches = batches  # 数据集
        self.device = device  # 设备
        self.model_name = model_name  # 模型名称
        self.n_batches = len(batches) // batch_size  # 批次的数量
        self.residue = (
            False if len(batches) % self.n_batches == 0 else True
        )  # 是否有剩余数据

    def _to_tensor(self, datas):
        """
        将数据转化为张量的函数

        参数：
        datas:数据

        返回：
        (x, seq_len), y
        """
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        # 若为BERT模型，返回(x, seq_len, mask), y；
        if self.model_name == "bert":
            mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
            return (x, seq_len, mask), y
        # 若为TextCNN模型，返回(x, seq_len), y
        if self.model_name == "textCNN":
            return (x, seq_len), y

    def __next__(self):
        """
        获取下一个批次的样本
        """
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size : len(self.batches)]
            self.index += 1
            return self._to_tensor(batches)
        elif self.index == self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[
                self.index * self.batch_size : (self.index + 1) * self.batch_size
            ]
            self.index += 1
            return self._to_tensor(batches)
    
    def __iter__(self):
        """
        返回迭代器本身
        """
        return self
    
    def __len__(self):
        """
        返回迭代器的长度
        """
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

def build_iterator(dataset,config):
    """
    根据配置信息构建数据集迭代器

    返回:
    数据迭代器对象
    """
    iter = DatasetIterater(dataset,config.batch_size,config.device,config.model_name)
    return iter

def get_time_dif(start_time):
    """
    获取已使用时间的时间差
    """
    end_time = time.time()
    time_dif = end_time - start_time
    # 将时间差转换为整数秒，并返回时间差对象
    return timedelta(seconds=int(round(time_dif)))

def build_dataset(config):
    """
    根据配置信息构建模型训练所需的数据集。

    参数：
    - config (object): 配置信息对象，包含有关数据集和模型的相关参数。

    返回：
    - train, dev, test (tuple): 包含三个元组，分别是训练集、验证集和测试集。
    """
    def load_dataset(path, pad_size=32):
        """
        加载并处理单个数据集文件。

        参数：
        - path (str): 数据集文件路径。
        - pad_size (int): 填充到的序列长度，默认为32。

        返回：
        - contents (list): 包含处理后的数据的列表。
        """
        contents = []  # 用于存储处理后的数据的列表
        with open(path, "r", encoding="UTF-8") as f:
            for line in tqdm(f):  # 逐行遍历文件内容
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split("\t")  # 分割每行的内容和标签
                token = config.tokenizer.tokenize(content)  # 使用分词器对内容进行分词
                token = [CLS] + token  # 在分词结果前加入[CLS]标记
                seq_len = len(token)  # 计算序列长度
                mask = []  # 用于存储填充掩码

                token_ids = config.tokenizer.convert_tokens_to_ids(token)  # 将分词结果转换为词汇索引

                if pad_size:
                    if len(token) < pad_size:  # 如果序列长度小于设定的填充长度
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))  # 创建填充掩码
                        token_ids += [0] * (pad_size - len(token))  # 对词汇索引列表进行填充
                    else:
                        mask = [1] * pad_size  # 如果序列长度大于等于填充长度，填充掩码为全1
                        token_ids = token_ids[:pad_size]  # 对词汇索引列表进行截断
                        seq_len = pad_size  # 更新序列长度为填充长度
                # 将处理后的数据添加到contents列表
                contents.append((token_ids, int(label), seq_len, mask))
        return contents

    # 使用load_dataset函数加载训练集、验证集和测试集
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)

    # 返回三个元组，分别是训练集、验证集和测试集
    return train, dev, test

