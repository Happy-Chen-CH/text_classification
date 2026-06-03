# 1.导入依赖包
import time
import jieba
import fasttext

# 服务框架使用Flask, 导入工具包
from flask import Flask
from flask import request

# 2.实例化Flask对象
app = Flask(__name__)

# 加载自定义的停用词表
jieba.load_userdict('./data/stopwords.txt')

# 提供已经训练好的模型路径
model_save_path = './model/fasttext_model_1779338898.bin'

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
    input_text = ' '.join(jieba.lcut(text))

    # 执行模型的预测
    res = model.predict(input_text)
    predict_name = res[0][0]

    return predict_name


# 4.启动Flask服务
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
