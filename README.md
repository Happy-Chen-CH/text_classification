# 🗂️ Chinese News Text Classification | 中文新闻文本分类（bert预训练模型太大了传不上来，需要自己去官网下载哦）

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.x-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-orange.svg)](https://huggingface.co/)
[![FastText](https://img.shields.io/badge/FastText-0.9-yellowgreen.svg)](https://fasttext.cc/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

一个基于多种方法的中文新闻标题分类项目，支持**传统机器学习、深度学习、模型蒸馏**三条技术路线，覆盖从工程落地到学术实验的完整实践。
注：Bert_project和random_forest_and_fasttext这两个实现文本分类的方案都是经过我运行验证没问题的，但是经过蒸馏后的那个实现方案好像有点问题，欢迎各位大佬来修改！
这是csdn更具体的讲解地址：https://blog.csdn.net/2301_81954099/article/details/161665457?spm=1001.2014.3001.5502
这是我的csdn个人主页：https://blog.csdn.net/2301_81954099?spm=1000.2115.3001.5343

## ✨ 特性

- 🔬 **四种方法，三套方案**：覆盖传统 ML (TF-IDF + 随机森林)、浅层神经网络 (FastText)、深层预训练模型 (BERT) 以及知识蒸馏 (BERT → TextCNN)
- 📊 **大规模中文语料**：18 万条训练数据 + 1 万验证 + 1 万测试，10 个新闻类别
- 🚀 **模型量化部署**：PyTorch 动态量化使 BERT 模型体积从 390MB 压缩至 146MB
- 🌐 **Flask API 服务**：提供开箱即用的 RESTful 推理接口
- 📦 **知识蒸馏实践**：Hinton 知识蒸馏方法，将大模型知识迁移至轻量模型
- 🖥️ **多设备支持**：兼容 CUDA / MPS (Apple Silicon) / CPU，自动检测最佳设备

## 📁 项目结构

```
text_classification/
├── randomforest_and_fasttext/      # 传统机器学习方案
│   ├── data/                       # 数据集 & 停用词表
│   │   ├── train.txt               # 训练集 (18万条)
│   │   ├── dev.txt                 # 验证集 (1万条)
│   │   ├── test.txt                # 测试集 (1万条)
│   │   └── class.txt               # 10个类别名称
│   ├── model/                      # 训练好的 FastText 模型
│   ├── analysis.py                 # 数据探索性分析 (EDA)
│   ├── preprocess.py               # 数据预处理 (FastText 格式)
│   ├── random_forest.py            # TF-IDF + 随机森林
│   ├── FastText-Train.py           # FastText 训练 (基础版)
│   ├── FastText-Train2.py          # FastText 训练 (增强 autotune)
│   ├── app.py                      # Flask 推理服务
│   └── test.py                     # API 客户端测试脚本
│
├── Bert_project/                   # BERT 深度方案
│   ├── data/
│   │   ├── bert_pretrain/          # 中文 BERT 预训练权重
│   │   │   ├── pytorch_model.bin   # 预训练模型 (393MB)
│   │   │   ├── bert_config.json    # 模型配置
│   │   │   └── vocab.txt           # 词表 (21128 tokens)
│   │   └── data1/                  # 训练/验证/测试数据
│   ├── src/
│   │   ├── models/bert.py          # BERT 模型定义
│   │   ├── run.py                  # 标准训练入口
│   │   ├── run1.py                 # 训练 + 动态量化
│   │   ├── train_eval.py           # 训练/评估/测试逻辑
│   │   ├── utils.py                # 数据集构建 & 迭代器
│   │   ├── predict.py              # 命令行单条推理
│   │   ├── app.py                  # Flask 推理服务 (量化模型)
│   │   ├── demo.py                 # API 客户端示例
│   │   └── saved_dict/             # 训练好的模型权重
│   │       ├── bert.pt             # 原始模型 (390MB)
│   │       └── bert_quantized.pt   # 量化模型 (146MB)
│
└── Bert_distil/                    # 知识蒸馏方案
    ├── data/
    │   ├── bert_pretrain/          # 中文 BERT 预训练权重 (教师)
    │   └── data/                   # 训练数据 + char级词表
    ├── src/
    │   ├── models/
    │   │   ├── bert.py             # BERT 教师模型
    │   │   └── textCNN.py          # TextCNN 学生模型
    │   ├── run.py                  # 训练入口 (--task trainbert / train_kd)
    │   ├── train_eval.py           # 标准训练 + 知识蒸馏训练
    │   └── utils.py                # 数据集构建
```

## 🧠 方法详解

### 1. TF-IDF + 随机森林

| 维度 | 说明 |
|------|------|
| **特征工程** | jieba 分词 → 去停用词 → TF-IDF 向量化 |
| **分类器** | scikit-learn RandomForestClassifier (多核并行) |
| **优势** | 训练快速，无需 GPU，可解释性强 |
| **文件** | [random_forest.py](randomforest_and_fasttext/random_forest.py) |

### 2. FastText

| 维度 | 说明 |
|------|------|
| **分词** | jieba 中文分词 |
| **模型** | fastText 监督学习，2-gram 词特征 |
| **调参** | autotune 自动超参搜索 (lr, dim, ws, epoch, minCount) |
| **规模** | 训练后模型约 370MB |
| **推理** | Flask REST API (`POST /v1/main_server/`) |
| **文件** | [FastText-Train2.py](randomforest_and_fasttext/FastText-Train2.py), [app.py](randomforest_and_fasttext/app.py) |

### 3. BERT (全量微调)

| 维度 | 说明 |
|------|------|
| **预训练模型** | 中文 BERT (12层, 768维, 21128 词表) |
| **分类头** | `BERT Encoder → Linear(768, 10)` |
| **序列长度** | 32 tokens (CLS 前置，padding/truncation) |
| **优化器** | AdamW (lr=5e-5, weight_decay=0.01) |
| **量化** | PyTorch 动态量化 (int8), 支持 QNNPACK / FBGEMM |
| **模型压缩** | 390MB → 146MB (量化后) |
| **设备** | CUDA / MPS (Apple Silicon) / CPU 自动适配 |
| **文件** | [run.py](Bert_project/src/run.py), [run1.py](Bert_project/src/run1.py) |

### 4. 知识蒸馏 (BERT → TextCNN)

| 维度 | 说明 |
|------|------|
| **教师模型** | 预训练中文 BERT (同方案3) |
| **学生模型** | TextCNN (char级 embedding, 多尺度卷积核 2/3/4, 256 filters) |
| **蒸馏损失** | KLDivLoss (T=2, α=0.8 软硬目标加权) |
| **教师输出** | 离线预计算，降低训练开销 |
| **学生参数** | embed_dim=300, dropout=0.5, lr=1e-3, 3 epochs |
| **文件** | [run.py](Bert_distil/src/run.py), [train_eval.py](Bert_distil/src/train_eval.py) |

## 📊 数据集

数据集来自中文新闻标题语料，共 **20 万条**标注数据：

| 集合 | 样本数 | 比例 |
|------|--------|------|
| 训练集 (train) | 180,000 | 90% |
| 验证集 (dev) | 10,000 | 5% |
| 测试集 (test) | 10,000 | 5% |

**10 个分类类别**：

| 编号 | 类别 | 英文 |
|------|------|------|
| 0 | 财经 | finance |
| 1 | 房产 | realty |
| 2 | 股票 | stocks |
| 3 | 教育 | education |
| 4 | 科技 | science |
| 5 | 社会 | society |
| 6 | 政治 | politics |
| 7 | 体育 | sports |
| 8 | 游戏 | game |
| 9 | 娱乐 | entertainment |

数据格式（tab 分隔）：
```
<sentence>\t<label_id>
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.x
- Transformers (HuggingFace)
- fastText
- jieba

### 安装依赖

```bash
# 创建虚拟环境 (推荐使用 conda)
conda create -n textcls python=3.8
conda activate textcls

# 安装依赖
pip install torch numpy pandas scikit-learn tqdm transformers fasttext jieba flask icecream
```

### 方式一：传统机器学习

```bash
# 1. 数据探索
cd randomforest_and_fasttext
python analysis.py

# 2. 预处理数据
python preprocess1.py

# 3. 训练 FastText
python FastText-Train2.py

# 4. 或训练随机森林
python random_forest.py

# 5. 启动 Flask 推理服务
python app.py

# 6. 测试 API
python test.py
```

### 方式二：BERT 全量微调

```bash
cd Bert_project/src

# 训练
python run.py --model bert

# 训练 + 量化
python run1.py

# 命令行单条推理
python predict.py

# 启动 Flask 推理服务
python app.py

# API 测试
python demo.py
```

### 方式三：知识蒸馏

```bash
cd Bert_distil/src

# 先训练 BERT 教师模型
python run.py --task trainbert

# 蒸馏训练 TextCNN 学生模型
python run.py --task train_kd
```

## 🔧 API 接口

两个子项目均提供 Flask RESTful 推理服务：

### FastText 服务 (端口 5000)

```python
import requests

url = "http://127.0.0.1:5000/v1/main_server/"
data = {"uid": "AI-20-202204", "text": "雷佳音获飞天奖"}
resp = requests.post(url, data=data)
print(resp.text)  # 输出预测类别
```

### BERT 量化模型服务

```python
import requests

url = "http://127.0.0.1:5000/predict"
data = {"text": "雷佳音获飞天奖"}
resp = requests.post(url, json=data)
print(resp.json())
```

## 🧪 模型对比

| 方法 | 参数量 | 模型大小 | 推理速度 | 适用场景 |
|------|--------|----------|----------|----------|
| Random Forest | — | — | ⚡ 极快 | 离线批量分类 |
| FastText | ~ | ~370MB | ⚡ 快 | 实时服务 (轻量) |
| BERT | 110M | 390MB | 🐢 慢 | 高精度场景 |
| BERT (量化) | 110M | 146MB | 🚀 较快 | 精度与速度折中 |
| TextCNN (蒸馏) | ~2M | 小 | ⚡ 快 | 移动端/边缘部署 |

## 📝 注意事项

1. **路径配置**：项目中部分代码包含硬编码的绝对路径，使用前请根据实际路径修改对应配置类中的 `data_path`、`save_path` 等变量
2. **模型文件**：预训练 BERT 权重 (`pytorch_model.bin`, ~393MB) 和训练好的模型权重 (`bert.pt`, `bert_quantized.pt`, fasttext `.bin` 文件) 体积较大，建议通过 Git LFS 管理或添加至 `.gitignore`
3. **设备兼容**：BERT 项目已针对 Apple Silicon (MPS) 做了适配，MPS 环境下自动跳过 CUDA 特有设置
4. **重复数据**：`Bert_project/data/data1/` 和 `Bert_distil/data/data/` 包含相同数据集，可按需统一为共享数据目录

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 License

MIT License

---

> 💡 **学习建议**：本项目适合作为 NLP 入门到进阶的实战案例，建议按 **FastText → BERT → 知识蒸馏** 的顺序学习，逐步理解从传统方法到深度学习再到模型压缩的完整技术演进路线。
