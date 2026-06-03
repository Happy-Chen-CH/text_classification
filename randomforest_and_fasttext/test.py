# 1.导入依赖包
import requests
import time

# 2. 向服务发送请求
# 定义请求的url地址和传入的数据
url = "http://127.0.0.1:5000/v1/main_server/"
# url = "http://0.0.0.0:5000/v1/main_server/"
# url = "http://198.18.0.1:5000/v1/main_server/"
data = {"uid": "AI-20-202204",
        "text": "雷佳音获飞天奖"}
# 计时
start_time = time.time()
res = requests.post(url, data=data)
# 获取处理时间
cost_time = time.time() - start_time

# 3.返回结果
print('输入文本:', data['text'])
print('分类结果:', res.text)
print('单条样本预测的耗时:', cost_time * 1000, 'ms')