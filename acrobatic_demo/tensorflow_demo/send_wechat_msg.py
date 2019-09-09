import itchat

# log in
itchat.auto_login(hotReload=True)

friends = []
msg = ""
for friend in friends:
    userfinfo = itchat.search_friends(friend)
    userid = userfinfo[0]["UserName"]
    itchat.send_msg(msg=msg, toUserName=userid)







import itchat

# 自动登录方法，hotReload=True可以缓存，不用每次都登录,但是第一次执行时会出现一个二维码，需要手机微信扫码登录
itchat.auto_login(hotReload=True)

# 搜索好友，search_friends("xxx"),其中"xxx"为好友昵称，备注或微信号不行
userfinfo = itchat.search_friends("智能群管家014")   # "智能群管家014"为好友昵称

# print(userfinfo)，获取userinfo中的UserName参数
userid = userfinfo[0]["UserName"]   # 获取用户id

# 调用微信接口发送消息
itchat.send("hello dear", userid)  # 通过用户id发送信息
# 或
itchat.send_msg(msg='hello dear', toUserName=userid)  # 发送纯文本信息



import datetime
import time
import requests
import json

github_url = "http://192.168.1.70:8080/robot/messageServlet2"

data = json.dumps({'name': 'test', 'phone': 139})

url = 'http://192.168.1.70:8080/robot/messageServlet2'
r = requests.post(url, data={'name': 'test', 'phone': 'some test repo'})

while 1:
    now = datetime.datetime.now()
    now_str = now.strftime('%Y/%m/%d %H:%M:%S')[11:]
    print('\r{}'.format(now_str), end='')
    r = requests.post(url, data={'name': 'test', 'phone': 139, 'content': '内容'})
    print(r.text.data)
    time.sleep(1)




import itchat

# itchat 微信官方教程：https://itchat.readthedocs.io/zh/latest/
# 微信登录
# 登录时如果断网，则此程序直接停止
# 启动热登录，并且生成 命令行 登录二维码
itchat.auto_login(hotReload=True, enableCmdQR=2)
# 保持心跳状态，防止自动退出登录
itchat.start_receiving()

# 获取群聊,注意群 必须保持到通讯录，否则可能会找不到群
itchat.get_chatrooms(update=True)
room = itchat.search_chatrooms('python')
if len(room) == 0:
    log.error('没有找到群信息')
else:
    try:
        iRoom = room[0]['UserName']
        # 发送消息
        result = itchat.send('send message', iRoom)
        try:
            if result['BaseResponse']['ErrMsg'] == '请求成功':
                print('send wechat success')
        except Exception as e:
            print('resolve wechat result fail,result is :{},error is {}'.format(result, e))
    except Exception as e:
        print('wechat send message fail,reason is :{} '.format(e))



import json
import requests
result = requests.get(
    url="https://qyapi.weixin.qq.com/cgi-bin/gettoken",
    params={'corpid': 'fg','corpsecret': '45'}
)
access_token = None
if result.status_code != 200:
    print('连接到服务器失败')
else:
    result_json = json.loads(result.text)
    if result_json['errcode'] != 0:
        print('响应结果不正确')
    else:
        access_token = result_json['access_token']
        print(access_token)

# 创建群聊
result = requests.post(
    url='https://qyapi.weixin.qq.com/cgi-bin/appchat/create?access_token={}'.format(access_token),
    data=json.dumps(
        {
            "name": "通知群",
            "owner": "user_name",
            "userlist": ["user_name", "user_name1", "user_name2"],
            "chatid": "secid"
        }
    )
)
print(result.text)

# 推送群聊信息
result = requests.post(
    url='https://qyapi.weixin.qq.com/cgi-bin/appchat/send?access_token={}'.format(access_token),
    data=json.dumps(
        {
            "chatid": "secid",
            "msgtype": "text",
            "text": {
                "content": "测试：你的快递已到\n请携带工卡前往邮件中心领取"
            },
            "safe": 0
        }
    )
)
print(result.text)

# 发送个人消息
result = requests.post(
    url='https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={}'.format(access_token),
    data=json.dumps(
        {
            "touser": "user_name",
            "msgtype": "text",
            "agentid": 23,
            "text": {
                "content": "你的快递已到，请携带工卡前往邮件中心领取。\n出发前可查看<a href=\"http://work.weixin.qq.com\">邮件中心视频实况</a>，聪明避开排队。"
            },
            "safe": 0
        }
    )
)
print(result.text)
