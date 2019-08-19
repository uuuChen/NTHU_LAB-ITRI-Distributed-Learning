from torch.autograd import Variable
import torch
import numpy as np
import time
from socket_ import Socket

while True:

    client = Socket(('localhost', 8080), False)
    client.connect()
    if client.is_right_conn(client_name='client_2'):
        msg = 'client_2'
        client.send(msg, 'msg')
    client.close()
    time.sleep(2)






