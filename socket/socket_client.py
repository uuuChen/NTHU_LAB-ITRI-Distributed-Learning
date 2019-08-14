#socket_client_tcp

# import socket
from socket import *
from torch.autograd import Variable
import torch
import numpy as np
import pickle

ip_port=('localhost',8080)
back_log=5
buffer_size=4096

tcp_client=socket(AF_INET,SOCK_STREAM)
tcp_client.connect(ip_port)

while True:
    msg=input('>>: ').strip()

    msg = np.array(int(msg))
    msg = torch.from_numpy(msg)
    msg = Variable(msg).float()
    msg = pickle.dumps(msg)
    if not msg:continue
    tcp_client.send(msg)
    print('客戶端已經發送訊息')
    data=tcp_client.recv(buffer_size)
    data = pickle.loads(data)
    print('收到服務端發來的訊息',data)

tcp_client.close()