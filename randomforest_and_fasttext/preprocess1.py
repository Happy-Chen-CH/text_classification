import jieba

#储存数据集中类别信息
id_to_label={}
#idx从0开始
idx=0
#打开数据集类别文件
with open('./data/class.txt','r',encoding='utf-8') as f1:
    for line in f1.readlines():
        line=line.strip('\n').strip()
        id_to_label[idx]=line
        idx+=1
#print('id_to_label:',id_to_label)

#储存数据集训练数据
train_data=[]
#打开训练集文件
with open('./data/train.txt','r',encoding='utf-8') as f2:
    for line in f2.readlines():
        #获得句子和标签id
        sentence,label_id=line.strip('\n').strip().split('\t')
        #把标签id转换成int
        label_id=int(label_id)
        label_name=id_to_label[label_id]
        new_label='__label__'+label_name
        sent_char=' '.join(jieba.lcut(sentence))
        #拼接成fasttext格式
        new_sentence=new_label+' '+sent_char
        train_data.append(new_sentence)
#print('train_data前五:',train_data[:5])

#保存新的数据到文件夹中
with open('./data/train_fast1.txt','w',encoding='utf-8') as f3:
    for data in train_data:
        f3.write(data+'\n')
print('FastText训练数据预处理完成')


