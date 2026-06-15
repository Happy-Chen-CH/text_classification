# 🗂️ Chinese News Text Classification · 中文新闻文本分类

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.x-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-orange.svg)](https://huggingface.co/)
[![FastText](https://img.shields.io/badge/FastText-0.9-yellowgreen.svg)](https://fasttext.cc/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E.svg)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-lightgrey.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

一个基于多种方法的中文新闻标题分类项目，覆盖**传统机器学习、浅层神经网络、深度预训练模型、知识蒸馏**四条技术路线，提供从数据处理、模型训练、量化部署到 API 服务的完整实践。

注：`Bert_project` 和 `randomforest_and_fasttext` 这两个实现文本分类的方案都是经过运行验证没问题的，但是经过蒸馏后的那个实现方案好像有点问题，欢迎各位大佬来修改！

> 📖 CSDN 详细讲解：[点击查看](https://blog.csdn.net/2301_81954099/article/details/161665457?spm=1001.2014.3001.5502)　|　👤 CSDN 个人主页：[Happy-Chen-CH](https://blog.csdn.net/2301_81954099?spm=1000.2115.3001.5343)

> ⚠️ **注意**：`Bert_project` 和 `randomforest_and_fasttext` 两个方案已经过完整运行验证；`Bert_distil`（知识蒸馏）方案可能存在一些问题，欢迎各位开发者提交 PR 修复！

---

## 📖 目录

- [项目简介](#-项目简介)
- [方法概览](#-方法概览)
- [项目结构](#-项目结构)
- [数据集](#-数据集)
- [环境要求](#-环境要求)
- [快速开始](#-快速开始)
  - [方案一：TF-IDF + 随机森林](#方案一tf-idf--随机森林)
  - [方案二：FastText](#方案二fasttext)
  - [方案三：BERT 全量微调](#方案三bert-全量微调)
  - [方案四：知识蒸馏 (BERT → TextCNN)](#方案四知识蒸馏-bert--textcnn)
- [API 接口](#-api-接口)
- [模型加载说明](#-模型加载说明)
- [模型对比](#-模型对比)
- [注意事项](#-注意事项)
- [贡献](#-贡献)
- [License](#-license)

---

## 📌 项目简介

本项目对中文新闻标题进行 **10 分类**，共 20 万条标注数据。实现了四种不同复杂度的方法，适合作为 NLP 从入门到进阶的实战案例：

| 方案 | 方法 | 核心技术 | 适用场景 |
|------|------|----------|----------|
| **方案一** | TF-IDF + 随机森林 | jieba 分词 → TF-IDF → RandomForest | 离线批量分类，无需 GPU |
| **方案二** | FastText | jieba 分词 + fastText 监督学习 | 轻量级实时服务 |
| **方案三** | BERT 微调 | HuggingFace `bert-base-chinese` + 动态量化 | 高精度场景 |
| **方案四** | 知识蒸馏 | BERT（教师）→ TextCNN（学生），KL 散度蒸馏 | 移动端 / 边缘部署 |

---

## 🧠 方法概览

### 1. TF-IDF + 随机森林

```
原始文本 → jieba 分词 → 去停用词 → TF-IDF 向量化 → RandomForestClassifier
```

- **特征工程**：jieba 中文分词后去除停用词，使用 `TfidfVectorizer` 将文本转为稀疏向量
- **分类器**：scikit-learn `RandomForestClassifier`，开启多核并行 (`n_jobs=-1`)
- **评估**：准确率 (Accuracy)、精确率 (Precision)、召回率 (Recall)、F1-Score
- **文件**：[`randomforest_and_fasttext/random_forest.py`](randomforest_and_fasttext/random_forest.py)

### 2. FastText

```
原始文本 → jieba 分词 → FastText 格式 (__label__X text) → fasttext.train_supervised
```

- **分词**：jieba 中文分词，空格拼接
- **模型**：fastText 官方库监督学习，使用 2-gram 词特征
- **调参**：`autotuneDuration=300s` 自动搜索最优超参数（学习率、向量维度、窗口大小、epoch 等）
- **推理**：Flask REST API 服务
- **文件**：[`randomforest_and_fasttext/FastText-Train2.py`](randomforest_and_fasttext/FastText-Train2.py)、[`randomforest_and_fasttext/app.py`](randomforest_and_fasttext/app.py)

### 3. BERT 全量微调

```
原始文本 → BertTokenizer → [CLS] + token_ids + mask → BertModel (12层, 768维) → Linear(768, 10)
```

| 参数 | 值 |
|------|-----|
| 预训练模型 | `bert-base-chinese`（从 HuggingFace Hub 在线加载） |
| 序列长度 | 32 tokens |
| 批次大小 | 128 |
| 学习率 | 5e-5 |
| 优化器 | AdamW (weight_decay=0.01) |
| Epoch | 1 |

- **模型来源**：预训练 BERT 通过 HuggingFace Hub 在线拉取，首次运行自动下载并缓存至 `~/.cache/huggingface/`
- **量化**：PyTorch 动态量化 (int8)，将 Linear 层量化为 int8，模型体积约 390MB → 146MB，支持 QNNPACK / FBGEMM 引擎
- **设备适配**：自动检测 CUDA / MPS (Apple Silicon) / CPU
- **推理**：提供命令行推理脚本和 Flask API 服务
- **文件**：[`Bert_project/src/run.py`](Bert_project/src/run.py)、[`Bert_project/src/run1.py`](Bert_project/src/run1.py)、[`Bert_project/src/predict.py`](Bert_project/src/predict.py)、[`Bert_project/src/app.py`](Bert_project/src/app.py)

### 4. 知识蒸馏 (BERT → TextCNN)

```
┌─────────────────────────────────────────────────────┐
│                    训练流程                          │
│                                                     │
│  BERT (教师) ──→ 预计算软目标 (offline)              │
│                          │                          │
│                          ▼                          │
│  TextCNN (学生) ←── KLDivLoss(软目标) + CrossEntropy(硬目标)  │
│       ▲                                             │
│       └── α=0.8 (软损失权重), T=2 (温度)              │
└─────────────────────────────────────────────────────┘
```

- **教师模型**：BERT `bert-base-chinese`（12 层，768 维隐藏层）
- **学生模型**：TextCNN（字符级嵌入，3 种卷积核 2/3/4，各 256 个滤波器）
  - `embed_dim=300`, `dropout=0.5`, `lr=1e-3`, 3 epochs
- **蒸馏方法**：Hinton 知识蒸馏，KLDivLoss + CrossEntropyLoss 联合优化
- **优化策略**：教师输出离线预计算，避免重复前向传播
- **文件**：[`Bert_distil/src/run.py`](Bert_distil/src/run.py)、[`Bert_distil/src/train_eval.py`](Bert_distil/src/train_eval.py)

---

## 📁 项目结构

```
text_classification/
│
├── randomforest_and_fasttext/          # 传统机器学习方案
│   ├── data/                           # 数据集
│   │   ├── train.txt                   # 训练集 (18万条，tab分隔)
│   │   ├── dev.txt                     # 验证集 (1万条)
│   │   ├── test.txt                    # 测试集 (1万条)
│   │   ├── class.txt                   # 10个类别名称
│   │   ├── stopwords.txt               # 中文停用词表
│   │   ├── train_new.csv               # jieba分词后的训练数据
│   │   ├── train_fast.txt              # FastText格式训练数据 (v1)
│   │   ├── train_fast1.txt             # FastText格式训练数据 (v2)
│   │   ├── dev_fast.txt                # FastText格式验证数据
│   │   └── test_fast.txt               # FastText格式测试数据
│   ├── model/                          # 训练好的 FastText 模型 (gitignored)
│   ├── analysis.py                     # 数据探索性分析 (EDA)
│   ├── preprocess.py                   # 数据预处理 → FastText格式 v1
│   ├── preprocess1.py                  # 数据预处理 → FastText格式 v2
│   ├── random_forest.py                # TF-IDF + 随机森林训练 & 评估
│   ├── FastText-Train.py               # FastText 训练 (基础版)
│   ├── FastText-Train2.py              # FastText 训练 (autotune超参搜索)
│   ├── app.py                          # Flask 推理服务
│   └── test.py                         # API 客户端测试
│
├── Bert_project/                       # BERT 深度方案
│   ├── data/
│   │   ├── bert_pretrain/              # BERT 配置文件 (模型权重从HuggingFace在线加载)
│   │   │   ├── bert_config.json        # BERT 模型配置 (12层, 768维)
│   │   │   └── vocab.txt               # 词表 (21128 tokens)
│   │   └── data1/                      # 训练/验证/测试数据
│   │       ├── train.txt
│   │       ├── dev.txt
│   │       ├── test.txt
│   │       └── class.txt
│   └── src/
│       ├── models/
│       │   └── bert.py                 # BERT 模型定义 & 配置类
│       ├── run.py                      # 标准训练入口 (--model bert)
│       ├── run1.py                     # 训练 + 动态量化 + 测试
│       ├── train_eval.py               # train() / evaluate() / test()
│       ├── utils.py                    # build_dataset() / DatasetIterater
│       ├── predict.py                  # 命令行单条推理
│       ├── app.py                      # Flask 推理服务 (量化模型)
│       ├── demo.py                     # API 客户端示例
│       ├── saved_dic/                  # 微调模型保存目录 (gitignored)
│       └── saved_dic1/                 # 量化模型保存目录 (gitignored)
│
├── Bert_distil/                        # 知识蒸馏方案
│   ├── data/
│   │   ├── bert_pretrain/              # BERT 配置文件 (模型权重从HuggingFace在线加载)
│   │   └── data/                       # 训练数据
│   │       ├── train.txt
│   │       ├── dev.txt
│   │       ├── test.txt
│   │       └── class.txt
│   └── src/
│       ├── models/
│       │   ├── bert.py                 # BERT 教师模型
│       │   └── textCNN.py              # TextCNN 学生模型 (char级嵌入)
│       ├── run.py                      # 训练入口 (--task trainbert / train_kd)
│       ├── train_eval.py               # train() / train_kd() / test() / evaluate()
│       ├── utils.py                    # build_dataset() / build_dataset_CNN()
│       └── saved_dic/                  # 模型保存目录 (gitignored)
│
├── .gitignore                          # Git 忽略规则 (含模型文件、checkpoint等)
└── README.md                           # 本文件
```

---

## 📊 数据集

数据集来自中文新闻标题语料，共 **20 万条**标注数据：

| 集合 | 样本数 | 比例 |
|------|--------|------|
| 训练集 (train) | 180,000 | 90% |
| 验证集 (dev) | 10,000 | 5% |
| 测试集 (test) | 10,000 | 5% |

### 10 个分类类别

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

### 数据格式

采用 **tab 分隔** 的纯文本格式：

```
<sentence>\t<label_id>
```

示例：
```
雷佳音获飞天奖	9
高考即将开始	3
```

---

## 🔧 环境要求

| 依赖 | 版本 | 用途 |
|------|------|------|
| Python | 3.8+ | 运行环境 |
| PyTorch | 1.x+ | BERT / TextCNN 深度学习框架 |
| Transformers | 4.x+ | HuggingFace BERT 模型加载 |
| fastText | 0.9.x | FastText 训练与推理 |
| scikit-learn | 1.x+ | TF-IDF / 随机森林 |
| jieba | 0.42+ | 中文分词 |
| Flask | 2.x+ | REST API 服务 |
| tqdm | 4.x+ | 进度条显示 |
| NumPy / Pandas | — | 数据处理 |

### 安装

```bash
# 创建虚拟环境 (推荐 conda)
conda create -n textcls python=3.8
conda activate textcls

# 安装核心依赖
pip install torch numpy pandas scikit-learn tqdm transformers fasttext jieba flask

# 可选依赖
pip install icecream  # 随机森林脚本调试输出
```

---

## 🔗 数据流与脚本依赖

项目中多个脚本之间存在输入输出的依赖关系，运行前请确认执行顺序：

### 方案一、二（randomforest_and_fasttext/）脚本依赖

```
class.txt ───────────────────────────────────────────┐
train.txt ──┬── analysis.py ──→ train_new.csv ──→ random_forest.py
            ├── preprocess.py ──→ train_fast.txt ──→ FastText-Train.py
            └── preprocess1.py ──→ train_fast1.txt ──→ FastText-Train2.py
                                                         │
                                                    model/*.bin
                                                         │
                                                       app.py
```

| 上游脚本 | 输出文件 | 下游脚本（消费者） |
|----------|----------|-------------------|
| `analysis.py` | `train_new.csv` | `random_forest.py` |
| `preprocess.py` | `train_fast.txt` | `FastText-Train.py` |
| `preprocess1.py` | `train_fast1.txt` | `FastText-Train2.py` |
| `FastText-Train2.py` | `model/fasttext_model_*.bin` | `app.py`（自动匹配最新文件） |

> ⚠️ `preprocess.py` 和 `preprocess1.py` 功能几乎相同，但输出文件名不同，对应的训练脚本也不同。**推荐使用 `preprocess1.py` + `FastText-Train2.py`**（带 autotune 超参搜索）。

### 方案三、四（Bert_project / Bert_distil）脚本依赖

```
HuggingFace Hub (bert-base-chinese) ──→ 首次运行时自动下载，缓存至 ~/.cache/huggingface/

Bert_project:
  run.py ──→ saved_dic/bert.pt ──→ predict.py / app.py / run1.py
  run1.py ──→ saved_dic1/bert_quantized.pt ──→ app.py

Bert_distil:
  run.py --task trainbert ──→ ../Bert_project/src/saved_dic/bert.pt（共享）
  run.py --task train_kd  ──→ saved_dic/textCNN.pt（需要先有教师模型）
```

> ⚠️ 方案三的步骤 2（`run1.py`）依赖步骤 1（`run.py`）产出的 `bert.pt`。方案四的蒸馏训练（`train_kd`）依赖先完成教师模型训练（`trainbert`）。

---

## 🚀 快速开始

### 方案一：TF-IDF + 随机森林

```bash
cd randomforest_and_fasttext

# 1. 数据探索分析（可选，同时会生成 train_new.csv 供第3步使用）
python analysis.py

# 2. 数据预处理（生成 train_new.csv，如果已通过第1步生成则可跳过）
python preprocess.py

# 3. 训练 & 评估随机森林（读取 train_new.csv）
python random_forest.py
```

**输出**：终端打印准确率 (Accuracy) 及训练耗时。

> 💡 `analysis.py` 和 `preprocess.py` 都会生成 `train_new.csv`，只需运行其中一个即可。

---

### 方案二：FastText

```bash
cd randomforest_and_fasttext

# 1. 数据预处理（生成 train_fast1.txt）
python preprocess1.py

# 2. 训练 FastText 模型（带 autotune 超参搜索，约 5 分钟）
python FastText-Train2.py

# 3. 启动 Flask 推理服务（自动匹配 model/ 目录下最新的模型文件）
python app.py

# 4. 另开终端，测试 API
python test.py
```

**输出**：
- 模型文件保存在 `model/fasttext_model_<timestamp>.bin`
- Flask 服务监听 `http://127.0.0.1:5000`

---

### 方案三：BERT 全量微调

```bash
cd Bert_project/src

# 1. 标准训练（首次运行会自动从 HuggingFace 下载 bert-base-chinese，约需下载 400MB）
python run.py --model bert

# 2. 训练 + 动态量化 + 测试（需要步骤1产出的 bert.pt）
python run1.py

# 3. 命令行单条推理（需要步骤1产出的 bert.pt）
python predict.py

# 4. 启动 Flask 推理服务（自动加载量化模型，如无则回退到 bert.pt）
python app.py

# 5. API 测试
python demo.py
```

**输出**：
- 微调模型：`saved_dic/bert.pt`（gitignored）
- 量化模型：`saved_dic1/bert_quantized.pt`（gitignored）
- 训练过程实时显示 Train Loss / Train Acc / Val Loss / Val Acc
- 测试集输出 Accuracy、Precision/Recall/F1-Score、混淆矩阵

---

### 方案四：知识蒸馏 (BERT → TextCNN)

```bash
cd Bert_distil/src

# 第1步：训练 BERT 教师模型（产出的 bert.pt 保存在 Bert_project/src/saved_dic/）
python run.py --task trainbert

# 第2步：蒸馏训练 TextCNN 学生模型（需先完成第1步，读取教师模型产出）
python run.py --task train_kd
```

**蒸馏流程**：
1. `trainbert` — 训练 BERT 教师模型，保存至 `Bert_project/src/saved_dic/bert.pt`
2. `train_kd` — 加载教师模型，预计算教师软目标，蒸馏训练 TextCNN，学生模型保存至 `saved_dic/textCNN.pt`

---

## 🌐 API 接口

两个子项目均提供基于 Flask 的 RESTful 推理服务。

### FastText 服务

**启动**：
```bash
cd randomforest_and_fasttext
python app.py
```

**接口**：`POST http://127.0.0.1:5000/v1/main_server/`

**请求格式** (`application/x-www-form-urlencoded`)：

| 参数 | 类型 | 说明 |
|------|------|------|
| `uid` | string | 用户/请求标识 |
| `text` | string | 待分类的中文文本 |

**调用示例**：
```python
import requests

url = "http://127.0.0.1:5000/v1/main_server/"
data = {"uid": "test-001", "text": "雷佳音获飞天奖"}
resp = requests.post(url, data=data)
print(resp.text)  # → __label__entertainment
```

### BERT 量化模型服务

**启动**：
```bash
cd Bert_project/src
python app.py
```

**接口**：`POST http://127.0.0.1:5000/v1/main_server/`

**请求格式** (`application/x-www-form-urlencoded`)：

| 参数 | 类型 | 说明 |
|------|------|------|
| `uid` | string | 用户/请求标识 |
| `text` | string | 待分类的中文文本 |

**调用示例**：
```python
import requests

url = "http://127.0.0.1:5000/v1/main_server/"
data = {"uid": "test-001", "text": "高考即将开始"}
resp = requests.post(url, data=data)
print(resp.text)  # → education
```

---

## 🗂️ 模型加载说明

本项目**不再在仓库中存储大型模型文件**，改为从线上自动拉取：

| 模型 | 加载方式 | 说明 |
|------|----------|------|
| **BERT 预训练模型** | HuggingFace Hub (`bert-base-chinese`) | 首次运行时自动下载，缓存至 `~/.cache/huggingface/` |
| **BERT 微调 checkpoint** | 本地 `torch.load` | 训练产出，已通过 `.gitignore` 排除 (`saved_dic/`) |
| **BERT 量化模型** | 本地 `torch.load` | 训练产出，已通过 `.gitignore` 排除 (`saved_dic1/`) |
| **FastText 模型** | 本地 `fasttext.load_model` | 训练产出，已通过 `.gitignore` 排除 (`*.bin`) |
| **TextCNN 词表** | 自动生成 | `vocab.pkl` 不存在时自动从训练数据构建 |
| **随机森林** | 内存训练 | 每次运行时重新训练，不持久化 |

> 💡 预训练 BERT 切换为在线加载后，项目体积大幅减小，可直接推送到 GitHub，无需 Git LFS。

---

## 🧪 模型对比

| 方法 | 模型体积 | 推理速度 | 精度 | 硬件需求 | 适用场景 |
|------|----------|----------|------|----------|----------|
| Random Forest | — | ⚡ 极快 | 中等 | CPU | 离线批量分类 |
| FastText | ~370MB | ⚡ 快 | 中高 | CPU | 轻量实时服务 |
| BERT | ~390MB | 🐢 慢 | ⭐ 高 | GPU 推荐 | 高精度场景 |
| BERT (量化) | ~146MB | 🚀 较快 | ⭐ 高 | CPU 可用 | 精度与速度折中 |
| TextCNN (蒸馏) | ~数MB | ⚡ 快 | 中高 | CPU | 移动端 / 边缘部署 |

---

## ⚠️ 注意事项

1. **首次运行需联网**：BERT 方案首次运行时，`transformers` 库会自动从 HuggingFace Hub 下载 `bert-base-chinese` 预训练模型（约 400MB），请确保网络畅通。下载后自动缓存，后续运行无需再次下载。

2. **路径配置**：各子项目的路径均基于脚本所在位置自动计算（`os.path.dirname`），一般无需手动修改。如需自定义数据路径，可编辑对应模型文件中的 `Config` 类。

3. **设备兼容**：
   - BERT 方案已针对 Apple Silicon (MPS) 做适配，MPS 环境下自动跳过 CUDA 特有 API
   - 量化引擎在 Apple Silicon 上使用 QNNPACK，在 x86 上使用 FBGEMM
   - 如遇量化失败，脚本会自动回退到原始模型继续运行

4. **数据集重复**：`Bert_project/data/data1/` 和 `Bert_distil/data/data/` 包含相同的数据集，可按需统一为共享数据目录。

5. **知识蒸馏方案**：该方案（`Bert_distil`）的蒸馏训练部分可能存在问题，欢迎调试并提交修复。

6. **模型文件**：所有大型模型文件（`*.pt`, `*.bin`, `*.pth`, `*.ckpt`, `*.safetensors`）和 `saved_dic*/` 目录已通过 `.gitignore` 排除出版本控制。

---

## ⚠️ 常见问题排查

### 1. 首次运行 BERT 方案时长时间卡住 / 下载失败

首次运行 BERT 相关脚本时，`transformers` 库会自动从 HuggingFace Hub 下载 `bert-base-chinese` 预训练模型（约 400MB）。

```bash
# 如果下载缓慢，可设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或在 Python 中指定镜像
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

模型下载后缓存于 `~/.cache/huggingface/`，后续运行无需再次下载。

### 2. `RuntimeError: CUDA out of memory`

GPU 显存不足。可以尝试：
- 减小 `batch_size`（在对应模型文件的 `Config` 类中修改，默认为 128）
- 缩短 `pad_size`（默认为 32）
- 切换到 CPU 训练（BERT 训练较慢，但可行）

### 3. `AssertionError: Torch not compiled with CUDA enabled`

在 Mac 或没有 CUDA 的机器上运行时的正常现象。本项目已对 MPS (Apple Silicon) 和 CPU 做了兼容处理：
- `Bert_project`：自动检测设备，CUDA 调用会被跳过
- `Bert_distil`：同样已添加 MPS 保护，无 CUDA 时自动跳过
- `randomforest_and_fasttext`：不使用 GPU，无需担心

### 4. `FileNotFoundError: 未找到FastText模型文件`

FastText Flask 服务（`app.py`）需要先训练生成模型文件。请先运行：

```bash
python preprocess1.py    # 生成训练数据
python FastText-Train2.py  # 训练模型（产出 model/fasttext_model_*.bin）
python app.py             # 再启动服务
```

### 5. BERT 推理脚本报错找不到 `bert.pt`

`predict.py` 和 `app.py` 需要先运行训练（`run.py`）产出微调后的模型 checkpoint。执行顺序：

```bash
python run.py --model bert   # 先训练，产出 saved_dic/bert.pt
python predict.py            # 再推理
python app.py                # 或启动 Flask 服务
```

### 6. 知识蒸馏方案 (`Bert_distil`) 效果不佳

该方案的字符级 TextCNN 词表构建函数此前存在 bug（词表只从第一行构建），现已修复。如果之前运行过该方案且效果不好，建议：

```bash
# 删除旧词表缓存，触发重新构建
rm Bert_distil/data/data/vocab.pkl
# 重新训练
python run.py --task trainbert
python run.py --task train_kd
```

### 7. `ModuleNotFoundError: No module named 'xxx'`

确保已在正确的 conda 环境中并安装了全部依赖：

```bash
conda activate textcls
pip install torch numpy pandas scikit-learn tqdm transformers fasttext jieba flask
```

### 8. Mac Apple Silicon (M1/M2/M3) 相关

- **量化引擎**：自动使用 QNNPACK（FBGEMM 在 ARM Mac 上不可用）
- **MPS 加速**：`Bert_project` 的 MPS 支持已被注释（稳定性考虑），默认使用 CPU；如需启用可取消 `bert.py` Config 中 MPS 相关代码的注释
- 训练速度较慢属于正常现象，建议减小 `batch_size` 和 `epoch` 数进行实验

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

- 🐛 发现 Bug → 提交 Issue
- 🔧 修复问题 → Fork → PR
- 💡 新功能 / 改进 → 先开 Issue 讨论

特别欢迎针对 `Bert_distil` 知识蒸馏方案的修复和改进。

---

## 📄 License

本项目采用 [MIT License](LICENSE) 开源。

---

> 💡 **学习路线建议**：推荐按 **FastText → BERT 微调 → 知识蒸馏** 的顺序逐步深入，从浅层模型理解分类基础，到深度预训练模型掌握微调范式，最后通过知识蒸馏实践模型压缩的完整技术链路。
