
from socket_ import *

server = Socket(('localhost', 8080), True)

server.accept()

print(server.addr[0])





