import torch
from tqdm import tqdm
import time
from datetime import timedelta


def build_dataset(config):
    """
    根据配置信息构建模型所需要的数据集

    参数：
    - config: 配置信息对象，包含有关数据集和模型的相关参数

    返回：
    - train, dev, test (tuple): 包含三个元组，分别是训练集、验证集和测试集。
    """
    PAD, CLS = "[PAD]", "[CLS]"

    def load_dataset(path, pad_size=32):
        """
        加载并处理单个数据集文件

        参数：
        - path(str): 数据集文件的路径
        - pad_size(int): 填充到序列的长度

        返回：
        - contents(list): 处理后的数据集列表
        """
        contents = []  # 用于储存处理过后的数据的列表
        with open(path, "r", encoding="utf-8") as f:
            for line in tqdm(f):  # 逐行遍历文件内容
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split("\t")  # 分割每行的句子和标签
                token = config.tokenizer.tokenize(content)  # 用分词器对内容进行分词
                token = [CLS] + token  # 添加CLS标记
                seq_len = len(token)  # 计算序列的长度
                mask = []  # 用于储存填充序列的mask

                token_ids = config.tokenizer.convert_tokens_to_ids(
                    token
                )  # 将分词结果转换为ID
                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (
                            pad_size - len(token)
                        )  # 创建mask填充掩码
                        token_ids += [0] * (pad_size - len(token))  # 对词汇索引列表填充
                    else:  # 如果序列的长度超过pad_size，则进行截断
                        mask = [1] * pad_size  # 创建mask填充掩码
                        token_ids = token_ids[:pad_size]  # 截取前pad_size个词
                        seq_len = pad_size  # 更新序列的长度
                # 将处理后的1行数据添加到content列表中
                contents.append((token_ids, int(label), seq_len, mask))
        return contents

    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return train, dev, test


# 构建数据迭代器的类
class DatasetIterater(object):
    def __init__(self, batches, batch_size, device, model_name):
        """
        数据集迭代器初始化函数
        参数：
        - batches: 样本列表
        - batch_size: 一批数据的大小
        - device: 设备对象，如'cpu'或'cuda'
        - model_name: 模型名称，用于选择不同的数据处理方式
        """
        self.batch_size = batch_size  # 一批次的大小
        self.batches = batches  # 总样本列表
        self.n_batches = len(batches) // batch_size  # 批次的数量
        self.model_name = model_name  # 模型名称
        self.residue = (
            False if len(batches) % self.n_batches == 0 else True
        )  # 判断是否有剩余的样本
        self.index = 0  # 当前批次的索引
        self.device = device  # 设备对象

    def _to_tensor(self, datas):
        """
        将数据(load_dataset返回的数据)转换为张量
        参数：
        - datas: 样本列表
        返回：
        - torch.LongTensor: 转换后的张量
        """
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)  # 创建词汇索引张量
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)  # 创建标签张量
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(
            self.device
        )  # 创建序列长度张量
        if self.model_name == "bert":
            mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
            return (x, seq_len, mask), y
        if self.model_name == "textCNN":
            return (x, seq_len), y

    def __next__(self):
        """
        获取下一批次的样本
        """
        if (
            self.residue and self.index == self.n_batches
        ):  # 如果有剩余的样本且当前索引等于总批次的数量
            batches = self.batches[self.index * self.batch_size : len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
        elif self.index >= self.n_batches:  # 如果当前索引大于等于总批次的数量
            self.index = 0
            raise StopIteration
        else:  # 否则，获取下一批次的样本
            batches = self.batches[
                self.index * self.batch_size : (self.index + 1) * self.batch_size
            ]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        """
        返回迭代器对象本身
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


def build_iterator(dataset, config):
    """
    根据配置信息构建数据集迭代器
    返回:数据迭代器对象
    """
    iter = DatasetIterater(dataset, config.batch_size, config.device, config.model_name)
    return iter


def get_time_dif(start_time):
    """
    获取已使用时间
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
