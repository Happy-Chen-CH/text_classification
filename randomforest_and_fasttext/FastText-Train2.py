import fasttext
import time

#获取训练集测试集和验证集的路径
train_path = "./data/train_fast1.txt"
test_path = "./data/test_fast.txt"
dev_path = "./data/dev_fast.txt"

# 开启模型训练
# autotuneValidationFile参数需要指定验证数据集所在的路径
# 它将在验证集是使用随机搜索的方法寻找最优的超参数
# 使用autotuneDuration参数可以控制随机搜索的时间, 默认是300秒.
# 根据不同的需求, 可以延长或者缩短时间.
# 调节的超参数包含这些内容:
# lr                         学习率 default 0.1
# dim                        词向量维度 default 100
# ws                         上下文窗口大小 default 5， cbow
# epoch                      epochs 数量 default 5
# minCount                   最低词频 default 5
# wordNgrams                 n-gram设置 default 1
# loss                       损失函数 {hs,softmax} default softmax
# minn                       最小字符长度 default 0
# maxn                       最大字符长度 default 0
# 我们设置verbose来观察超参数的值
# verbose: 该参数决定日志打印级别, 当设置为3, 可以将当前正在尝试的超参数打印出来
model = fasttext.train_supervised(input=train_path, #训练集路径
                                  autotuneValidationFile=dev_path, #验证数据集路径
                                  autotuneDuration=300, #随机搜索时间
                                  wordNgrams=2, #n-gram设置
                                  verbose=3 #输出日志级别
                                  )

#开启模型测试并打印
result = model.test(test_path)
print(result)

#获取当前时间，作为名称
time1=int(time.time())
model_save_path="./model/fasttext_model_%s.bin" % time1
model.save_model(model_save_path)
