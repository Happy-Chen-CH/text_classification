import torch
import numpy as np
from importlib import import_module

#定义特殊符号和类别映射
CLS='[CLS]'
id_to_name={0: 'finance', 1: 'realty', 2: 'stocks', 3: 'education', 4: 'science',
            5: 'society', 6: 'politics', 7: 'sports', 8: 'game', 9: 'entertainment'}

def inference(model,config,input_text,padding_size=32):
    """
    模型推理函数，
    参数：
    model:模型
    config:模型配置参数
    input_text:输入文本
    padding_size:句子长度
    """
    #对输入的句子进行分词和预处理
    content = config.tokenizer.tokenize(input_text)
    content = [CLS] + content
    seq_len = len(content)
    token_ids = config.tokenizer.convert_tokens_to_ids(content)

    #填充或补齐句子长度
    if seq_len<padding_size:
        mask = [1]*seq_len + [0]*(padding_size-seq_len)
        token_ids += [0]*(padding_size-seq_len)
    else:
        mask = [1]*padding_size
        token_ids=token_ids[:padding_size]
        seq_len = len(token_ids)

    #将处理后的文本转化为pytorch tensor
    x = torch.LongTensor(token_ids).to(config.device)
    seq_len = torch.LongTensor(seq_len).to(config.device)
    mask = torch.LongTensor(mask).to(config.device)

    #增加一维，表示batch_size为1
    x =x.unsqueeze(0)
    seq_len = seq_len.unsqueeze(0)
    mask = mask.unsqueeze(0)

    data=(x,seq_len,mask)
    #模型推理
    output=model(data)
    #获取预测结果id
    predict_result=torch.max(output,1)[1]

    return predict_result

if __name__ == '__main__':
    #加载BERT模型配置和模型
    model_name='bert'
    x = import_module('models.'+ model_name)
    config = x.Config()

    #设置随机种子
    np.random.seed(3)
    torch.manual_seed(3)

    #创建并加载BERT模型
    model=x.Model(config).to(config.device)
    model.load_state_dict(torch.load(config.save_path,map_location=config.device),strict=False)

    #待分析文本
    input_text = "高考即将开始"

    #进行模型推理
    res=inference(model,config,input_text)
    #获取类别名
    result=id_to_name[res.item()]

    print(result)