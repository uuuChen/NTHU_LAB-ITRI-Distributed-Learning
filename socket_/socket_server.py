from socket_ import *
from socket_args import *

server = Socket(socket_args=SERVER_SOCKET_ARGS)

server.accept()  # 服務端阻塞

while True:

    data = server.recv('msg')

    print(data)


