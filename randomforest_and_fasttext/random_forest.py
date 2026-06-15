"""
## 数据集分析  random_forest.py
# 1.导入依赖包
# 2.读取数据集
# 3.构建语料库
# 4.获取停用词
# 5.计算tfidf特征stopwords.txt
# 6.划分数据集
# 7.实例化模型
# 8.模型训练
# 9.模型评估
"""
import time

# 1.导入依赖包
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from icecream import ic
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

# 2.读取数据集
TRAIN_CORPUS = './data/train_new.csv'
STOP_WORDS = './data/stopwords.txt'
WORDS_COLUMN = 'words'
content = pd.read_csv(TRAIN_CORPUS)

# 3.构建语料库
corpus = content[WORDS_COLUMN].values

# 4.获取停用词
stopwords = open(STOP_WORDS).read().split()

# 5.计算tfidf特征stopwords.txt
tfidf = TfidfVectorizer(stop_words=stopwords)
text_vectors = tfidf.fit_transform(corpus)
targets=content['label']

# 6.划分数据集
X_train, X_test, y_train, y_test = train_test_split(text_vectors, targets, test_size=0.2, random_state=3)

# 7.实例化模型
estimator = RandomForestClassifier(n_jobs=-1)

# 8.模型训练
print('训练开始......')
start_time=time.time()
estimator.fit(X_train, y_train)

# 9.模型评估
y_predict = estimator.predict(X_test)
accuracy=accuracy_score(y_predict, y_test)
ic(accuracy)
print(time.time()-start_time)
