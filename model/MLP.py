import torch.nn.functional as F
import torch.nn as nn


class MLP(nn.Module):

     def __init__(self, input_node_nums, label_class_nums):

         super(MLP, self).__init__()

         self.fc1 = nn.Linear(input_node_nums, 1000)

         self.fc2 = nn.Linear(1000, 500)

         self.fc3 = nn.Linear(500, 125)

         self.fc4 = nn.Linear(125, label_class_nums)

     def forward(self, x):

         x = F.relu(self.fc1(x))

         x = F.relu(self.fc2(x))

         x = F.relu(self.fc3(x))

         x = self.fc4(x)

         return x


class Agent_MLP(nn.Module):

    def __init__(self, input_node_nums, conn_node_nums):

        super(Agent_MLP, self).__init__()

        self.fc1 = nn.Linear(input_node_nums, 1000)

        self.fc2 = nn.Linear(1000, conn_node_nums)

    def forward(self, x):

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        return x


class Server_MLP(nn.Module):

    def __init__(self, conn_node_nums, label_class_nums):

        super(Server_MLP, self).__init__()

        self.fc3 = nn.Linear(conn_node_nums, 125)

        self.fc4 = nn.Linear(125, label_class_nums)

    def forward(self, x):

        x = F.relu(self.fc3(x))

        x = self.fc4(x)

        return x












