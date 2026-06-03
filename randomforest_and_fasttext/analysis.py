import pandas as pd
from collections import Counter
import jieba


#读取文件，统计文件的长度，统计标签的个数，统计标签的占比
content=pd.read_csv('./data/train.txt',sep='\t',names=['sentence','label'])
print(content.head())
print(len(content))
count=Counter(content.label.values)
print(count)
print(len(count))
print('*'*50)

#统计标签的占比
values=[]
ratios=[]
total=0
for i,v in count.items():
    total+=v
    values.append(v)
    ratios.append(v/len(content) *100)
    print(f'{i}:{v/len(content) *100:.2f}%')
print(f'总样本数:{total}')
#print(values)
print('*'*50)

#统计文本长度，文本长度标准差，文本长度平均值
content['sentence_len']=content.sentence.apply(lambda x:len(x))
print(content.head())
length_mean=content.sentence_len.mean()
length_std=content.sentence_len.std()
print(f'平均长度:{length_mean:.2f}')
print(f'标准差:{length_std:.2f}')

#jieba分词
def jieba_cut(text):
    return list(jieba.cut(text))

content['words']=content.sentence.apply(jieba_cut)
print(content.head())
content['words']=content['sentence'].apply(lambda s: ' '.join(jieba_cut(s)))
content['words']=content['words'].apply(lambda s: ' '.join(s.split())[:30])
print(content.head())
content.to_csv('./data/train_new.csv')



