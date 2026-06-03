import jieba

id_to_label={}
idx=0
with open('./data/class.txt','r',encoding='utf-8') as f1:
    for line in f1.readlines():
        line=line.strip('\n').strip()
        id_to_label[idx]=line
        idx+=1
print(f'id_to_label:{id_to_label}')

train_data=[]
with open('./data/train.txt','r',encoding='utf-8') as f2:
    for line in f2.readlines():
        line=line.strip('\n').strip()
        sentence,label=line.split('\t')
        label_id=int(label)
        label_name=id_to_label[label_id]
        new_label='__label__'+label_name
        sent_char=' '.join(jieba.lcut(sentence))
        new_sentence=new_label+' '+sent_char
        train_data.append(new_sentence)
print(f'train_data:{train_data[:5]}')

with open('./data/train_fast.txt','w',encoding='utf-8') as f3:
    for line in train_data:
        f3.write(line+'\n')
print('FastText训练数据预处理完成')