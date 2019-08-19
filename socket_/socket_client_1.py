from torch.autograd import Variable
import torch
import numpy as np
import time
from socket_ import Socket

while True:
    client = Socket(('localhost', 8080), False)
    client.connect()
    if client.is_right_conn(client_name='client_1'):
        msg = 'client_1'
        client.send(msg, 'msg')
    client.close()
    time.sleep(2)

# client = Socket(('localhost', 8080), False)
#
# client.connect()
#
# while True:
#
#     if not client.is_right_conn(client_name='client_1'):
#         client.close()
#         time.sleep(10)
#         client = Socket(('localhost', 8080), False)
#         client.connect()
#
#     else:
#         msg = 'client_1'
#
#         client.send(msg, 'msg')






