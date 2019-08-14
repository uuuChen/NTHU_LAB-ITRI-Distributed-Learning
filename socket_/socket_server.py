from socket_ import *
from socket_args import *

server = Socket(socket_args=SERVER_SOCKET_ARGS)

while True:

    server.accept() #服務端阻塞

    data = server.send(1, 'train_args')

    while True:

        # server.send(data, 'server_msg')

    conn.close()


tcp_server.close()