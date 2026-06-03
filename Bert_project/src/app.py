import torch
from flask import Flask, request
from importlib import import_module
import numpy as np

# 定义BERT特殊符号
CLS = "[CLS]"

# 加载BERT模型配置和模型
model_name = "bert"
x = import_module("models." + model_name)
config = x.Config()
# 创建随机种子
np.random.seed(3)
torch.manual_seed(3)
# 创建模型
model = x.Model(config).to(config.device)
model.load_state_dict(torch.load(config.save_path, map_location="cpu"),strict=False)


# 推理函数，用于对输入的文本进行分类分析，与模型预测部分相同
def inference(model, config, input_text, padding_size=32):
    model.eval()
    with torch.no_grad():
    # 对输入的文本进行分词和预处理
        content = config.tokenizer.tokenize(input_text)
        content = [CLS] + content
        seq_len = len(content)
        token_ids = config.tokenizer.convert_tokens_to_ids(content)
        # 填充或截断到指定文本长度
        if seq_len < padding_size:
            mask = [1] * seq_len + [0] * (padding_size - seq_len)
            token_ids += [0] * (padding_size - seq_len)
        else:
            mask = [1] * padding_size
            token_ids = token_ids[:padding_size]
            seq_len = len(token_ids)
        # 将处理后的文本转化为Tensor形式
        x = torch.tensor(token_ids).to(config.device)
        seq_len = torch.tensor(seq_len).to(config.device)
        mask = torch.tensor(mask).to(config.device)
        # 增加一个维度，表示batch_size为1
        x = x.unsqueeze(0)
        seq_len = seq_len.unsqueeze(0)
        mask = mask.unsqueeze(0)
        data=(x, seq_len, mask)
        output = model(data)
        #获取预测结果并返回
        predict_result = torch.max(output, 1)[1]
        predict_result = predict_result.item()
        predict_result = config.class_list[predict_result]

        return predict_result

app = Flask(__name__)
@app.route('/v1/main_server/', methods=['POST'])
def main_server():
    #从post请求中获取用户id和文本数据
    uid = request.form.get('uid', '')
    input_text = request.form.get('text', '')
    #调用推理函数获取结果
    res=inference(model,config,input_text)
    print('获取到text=',repr(request.form.get('text')))
    return res

if __name__ == '__main__':
    app.run() 