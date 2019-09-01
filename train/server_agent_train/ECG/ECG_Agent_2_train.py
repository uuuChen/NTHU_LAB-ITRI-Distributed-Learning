# Model Imports
from model.MLP import *

# DataSet Imports
from data.data_args import *  # import data arguments

# Socket Imports
from train.agent import Agent

os.chdir('../../../')

# training settings
model_agent = Agent_MLP(input_node_nums=ECG_COMMON_ARGS['data_length'],
                        conn_node_nums=ECG_COMMON_ARGS['MLP_conn_node_nums'])

cur_agent_name = 'agent_2'

# ==================================
# LocalHost testing
# ==================================
server_host_port = ('localhost', 8081)

# ==================================
# LAN testing
# ==================================
# server_host_port = ('172.20.10.2', 8081)


if __name__ == '__main__':

    agent = Agent(model_agent, server_host_port, cur_agent_name)
    agent.start_training()







