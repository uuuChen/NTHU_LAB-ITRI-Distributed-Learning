#socket_server_tcp

# import socket
from socket import *
from torch.autograd import Variable
import torch
import numpy as np
import pickle

ip_port = ('localhost', 8080)
back_log = 5
buffer_size = 4096

tcp_server = socket(AF_INET,SOCK_STREAM)
tcp_server.bind(ip_port)
tcp_server.listen(back_log)

while True:
    print('服務端開始運行了')
    conn, addr = tcp_server.accept() #服務端阻塞
    print('雙向連結是',conn)
    print('客戶端地址',addr)

    while True:
        try:
            data = conn.recv(buffer_size)
            data = pickle.loads(data)

            print('客戶端發來的訊息是',data)
            data = pickle.dumps(data)
            conn.send(data)
        except Exception:
            break
    conn.close()


tcp_server.close()