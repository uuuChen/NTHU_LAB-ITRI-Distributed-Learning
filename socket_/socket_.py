
from socket import *
import pickle
import time

from logger import *

DEBUG = False

class Socket(Logger):

    def __init__(self, socket_args):

        Logger.__init__(self)

        self.__logger = self.get_logger(unique_name=__name__,
                                        debug=DEBUG)

        self.type = socket_args['type']

        self.host = socket_args['host']

        self.port = socket_args['port']

        self.back_log = socket_args['back_log']

        self.buffer_size = socket_args['buffer_size']

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

    def _send(self, data, data_name):

        data = pickle.dumps(data)

        try:
            if self.type == 'server':
                self.conn.sendall(data)

            else:
                self.socket.sendall(data)

        except Exception:
            self.__logger.error('"%s" Send "%s" Error !' % (self.type, data_name))
            raise

    def send(self, data, data_name):

        self.__logger.debug('----------------------------------------------------------------------')
        self.__logger.debug('SEND\n')

        # header setup and send
        header = {
            'data_name': data_name,
            'data_len': len(pickle.dumps(data))
        }
        self._send(header, data_name + '_header')

        self.__logger.debug('send data_len: %s' % header['data_len'])

        time.sleep(0.1)  # VERY IMPORTANT! otherwise it won’t pass because the interval is too close

        self._send(data, data_name)

        self.__logger.debug('"%s" Send "%s" Successfully !' % (self.type, data_name))

    def _recv(self, data_name, data_len, is_header=False):

        data = []

        left_data_len = data_len

        while True:

            # VERY IMPORTANT. Adjust "buf_size" to get just the right size of the data
            if left_data_len >= self.buffer_size:
                buf_size = self.buffer_size
            else:
                buf_size = left_data_len

            try:
                if self.type == 'server':
                    buf_data = self.conn.recv(buf_size)
                else:
                    buf_data = self.socket.recv(buf_size)

                if is_header:
                    data = [buf_data]
                    break

                data.append(buf_data)

                left_data_len -= len(buf_data)

                if left_data_len == 0:
                    break

            except Exception:
                self.__logger.error('"%s" Receive "%s" Error !' % (self.type, data_name))
                raise

        if is_header:
            self.__logger.debug('receive header len: ' + str(len(b"".join(data))))
        else:
            self.__logger.debug('receive data len: ' + str(len(b"".join(data))))

        data = pickle.loads(b"".join(data))

        return data

    def recv(self, data_name):

        self.__logger.debug('----------------------------------------------------------------------')
        self.__logger.debug('RECV\n')

        while True:

            # receive data header
            header_name = data_name + '_header'
            header = self._recv(header_name, data_len=self.buffer_size, is_header=True)

            self.__logger.debug('header: ' + str(header))

            # whether get the correct packet
            if data_name == header['data_name']:  # correct, break the while loop
                self.__logger.debug('matching header name')
                break

            else:  # incorrect, keep waiting
                self.__logger.error('"%s" Receive Wrong data. Expect to receive "%s" instead of "%s" !'
                                    % (self.type, data_name, header['data_name']))

        data = self._recv(data_name, data_len=header['data_len'])

        self.__logger.debug('"%s" Receive "%s" Successfully !' % (self.type, data_name))

        return data

    def accept(self):
        self.conn, self.addr = self.socket.accept()

    def close(self):
        self.conn.close()
        self.socket.close()



