
from socket import *
import pickle

from logger import *

DEBUG = True

class Socket(Logger):

    def __init__(self, socket_args):
        Logger.__init__(self)
        self.__logger = self.get_logger(unique_name=__name__,
                                        debug=DEBUG)
        self.type = socket_args['type']
        self.host = socket_args['host']
        self.port = socket_args['port']
        self.back_log = socket_args['back_log']
        self.max_buffer_size = socket_args['max_buffer_size']
        self.socket = self._get_socket()

    def _get_socket(self):
        if self.type == 'server':
            socket = self._get_server_socket()
        else:
            socket = self._get_client_socket()
        return socket

    def _get_server_socket(self):
        server = socket(AF_INET, SOCK_STREAM)
        server.bind((self.host, self.port))
        server.listen(self.back_log)
        self.__logger.debug('Server Starts Running...')
        return server

    def _get_client_socket(self):
        client = socket(AF_INET, SOCK_STREAM)
        client.connect((self.host, self.port))
        return client

    def send(self, data, data_name):
        msg = {
            'data': data,
            'data_name': data_name
        }
        msg = pickle.dumps(msg)
        try:
            if self.type == 'server':
                self.conn.sendall(msg)
            else:
                self.socket.sendall(msg)
        except Exception:
            self.__logger.error('"%s" Send "%s" Error !' % (self.type, data_name))
            raise
        self.__logger.debug('"%s" Send "%s" Successfully !' % (self.type, data_name))

    def recv(self, data_name):
        while True:
            try:
                if self.type == 'server':
                    msg = self.conn.recv(self.max_buffer_size)
                else:
                    msg = self.socket.recv(self.max_buffer_size)

            except Exception:
                self.__logger.error('"%s" Receive "%s" Error !' % (self.type, data_name))
                raise
            msg = pickle.loads(msg)
            rcv_data = msg['data']
            rcv_data_name = msg['data_name']
            if data_name == rcv_data_name:
                self.__logger.debug('"%s" Receive "%s" Successfully !' % (self.type, data_name))
                break
            else:
                self.__logger.debug('"%s" Receive Wrong data. Expect to receive "%s" instead of "%s" !'
                                    % (self.type, data_name, rcv_data_name))
        return rcv_data

    def accept(self):
        self.conn, self.addr = self.socket.accept()

    def close(self):
        self.conn.close()
        self.socket.close()



