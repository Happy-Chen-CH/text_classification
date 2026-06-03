import requests
import time

#定义url和传入的data
url = 'http://127.0.0.1:5000/v1/main_server/'
data={'uid':'AI','text':'高考即将开始 '}
start_time = time.time()
#向服务发送post请求
res=requests.post(url,data=data)
cost_time = time.time() - start_time

#打印返回的结果
print('文本类别：',res.text)
print('耗时：',cost_time*1000,'ms')