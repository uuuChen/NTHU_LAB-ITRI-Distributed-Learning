from torch.autograd import Variable
import torch
import numpy as np

from socket_ import Socket
from socket_args import *

client = Socket(socket_args=AGENT_SOCKET_ARGS)

while True:
    msg = input('>>: ').strip()
    msg = np.array(int(msg))
    msg = torch.from_numpy(msg)
    msg = Variable(msg).float()

    print(msg)
    client.send(msg, 'msg')




