# 1.导入依赖包
import time
import os
import glob
import jieba
import fasttext

# 服务框架使用Flask, 导入工具包
from flask import Flask
from flask import request

# 2.实例化Flask对象
app = Flask(__name__)

# 加载停用词表
with open('./data/stopwords.txt', 'r', encoding='utf-8') as f:
    stopwords = set(line.strip() for line in f if line.strip())

# 自动查找最新的模型文件
model_dir = './model/'
model_files = glob.glob(os.path.join(model_dir, 'fasttext_model_*.bin'))
if not model_files:
    raise FileNotFoundError(f"未找到FastText模型文件，请先在 {model_dir} 目录下训练模型")
model_save_path = max(model_files, key=os.path.getctime)
print(f'加载模型: {model_save_path}')

# 实例化fasttext对象, 并加载模型参数用于推断, 提供服务请求
model = fasttext.load_model(model_save_path)
print('FastText模型实例化完毕...')


# 3.设定投满分项目的服务的路由和请求方法
@app.route('/v1/main_server/', methods=["POST"])
def main_server():
    # 接收来自请求方发送的服务字段
    uid = request.form['uid']
    text = request.form['text']

    # 对请求文本进行处理, 因为前面加载的是基于分词的模型, 所以这里也要对text进行分词操作
    # 分词后去除停用词
    words = [w for w in jieba.lcut(text) if w not in stopwords]
    input_text = ' '.join(words)

    # 执行模型的预测
    res = model.predict(input_text)
    predict_name = res[0][0]
    # 去除 __label__ 前缀，只返回类别名
    if predict_name.startswith('__label__'):
        predict_name = predict_name[len('__label__'):]

    return predict_name


# 4.启动Flask服务
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
