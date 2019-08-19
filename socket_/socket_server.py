from socket_ import *
import time

server = Socket(('localhost', 8080), True)

last_time = time.time()
recv_client_name = 'client_2'

while True:

    if time.time() - last_time >= 10:
        last_time = time.time()
        if recv_client_name == 'client_1':
            recv_client_name = 'client_2'
        else:
            recv_client_name = 'client_1'
        print('change to %s' % recv_client_name)

    server.accept()  # 服務端阻塞

    if not server.is_right_conn(client_name=recv_client_name):
        continue

    data = server.recv('msg')

    print(data)





