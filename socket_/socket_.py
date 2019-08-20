
from socket import *
import pickle
import time

from logger import *

DEBUG = True

logger = Logger.get_logger(unique_name=__name__,
                           debug=DEBUG)

class Socket(Logger):

    def __init__(self, server_host_port, is_server, back_log=5, buffer_size=2048):

        Logger.__init__(self)

        # logger = self.get_logger(unique_name=__name__,
        #                          debug=DEBUG)

        self.__logger = logger

        if is_server:
            type_ = 'server'
        else:
            type_ = 'agent'
        self.type = type_

        self.is_server = is_server

        self.server_host_port = server_host_port

        self.back_log = back_log

        self.buffer_size = buffer_size

        self.socket = self._get_socket()

    def _get_socket(self):

        if self.is_server:
            socket = self._get_server_socket()

        else:
            socket = self._get_client_socket()

        return socket

    def _get_server_socket(self):

        server = socket(AF_INET, SOCK_STREAM)

        server.bind(self.server_host_port)

        server.listen(self.back_log)

        self.__logger.debug('Server Starts Running...')

        return server

    def _get_client_socket(self):

        client = socket(AF_INET, SOCK_STREAM)

        return client

    def _send(self, data, data_name):

        data = pickle.dumps(data)

        try:
            if self.is_server:
                self.conn.sendall(data)  # sendall() is a package of send(). It can automatically call send() when the
                                         # data has not been delivered completely.

            else:
                self.socket.sendall(data)

        except Exception:
            self.__logger.error('"%s" Send "%s" Error !' % (self.type, data_name))
            raise

    def send(self, data, data_name):

        """ When the sender "send", the header is sent first, and then data is sent.

        The contents of the header include:
            (1) data_name  (str): The name of the data to be passed, used to compare at the time of reception.
            (2) data_bytes (int): The total number of bytes of data to be sent.

        "data_name" is used to let the "receiver" confirm whether it is the data it wants. In addition, because the
        "receiver" can receive up to "self.buffer_size" bytes at a time, "data_bytes" is used to determine how much to
        receive each time, especially when receiving the last packet, the amount received must be correct, otherwise it
        is possible to receive packets of other data, which is very important.

        Args:
            data      (*)   : The information to be delivered. Not limited to the data type.
            data_name (str) : The name of the data to be sent, used to compare at the time of reception.

        Returns:
            None.

        """

        self.__logger.debug('----------------------------------------------------------------------')
        self.__logger.debug('SEND\n')

        # header setup and send
        header = {
            'data_name': data_name,
            'data_bytes': len(pickle.dumps(data))
        }
        self._send(header, data_name + '_header')

        self.__logger.debug('send data_bytes: %s' % header['data_bytes'])

        time.sleep(0.1)  # VERY IMPORTANT! otherwise it won’t pass because the interval is too close

        self._send(data, data_name)

        self.__logger.debug('"%s" Send "%s" Successfully !' % (self.type, data_name))

    def _recv(self, data_name, data_bytes, is_header=False):

        data = []

        left_data_bytes = data_bytes
        while True:

            # VERY IMPORTANT! Adjust "buf_size" to get just the right size of the data
            if left_data_bytes >= self.buffer_size:
                buf_size = self.buffer_size
            else:
                buf_size = left_data_bytes

            try:
                if self.is_server:
                    buf_data = self.conn.recv(buf_size)
                else:
                    buf_data = self.socket.recv(buf_size)

                if is_header:
                    data = [buf_data]
                    break

                data.append(buf_data)

                left_data_bytes -= len(buf_data)

                if left_data_bytes == 0:
                    break

            except timeout:

                self.__logger.error('TIMEOUT !!! "%s" Receive "%s" Error !' % (self.type, data_name))
                raise

        if is_header:
            self.__logger.debug('receive header bytes: ' + str(len(b"".join(data))))
        else:
            self.__logger.debug('receive data bytes: ' + str(len(b"".join(data))))

        data = pickle.loads(b"".join(data))

        return data

    def recv(self, data_name):

        """ When the receiver "recv", the header is received first, and then data is received.

        The contents of the header include:
            (1) data_name  (str): The name of the data to be passed, used to compare at the time of reception.
            (2) data_bytes (int): The total number of bytes of data to be sent.

        "data_name" is used to let the "receiver" confirm whether it is the data it wants. In addition, because the
        "receiver" can receive up to "self.buffer_size" bytes at a time, "data_bytes" is used to determine how much to
        receive each time, especially when receiving the last packet, the amount received must be correct, otherwise it
        is possible to receive packets of other data, which is very important.

        Args:
            data_name (str) : The name of the data to be received. If the data_name received by the header does not
                              match the data_name of the argument, continue to wait.

        Returns:
            data (*): The data sent by the sender. Not limited to the data type.

        """

        self.__logger.debug('----------------------------------------------------------------------')
        self.__logger.debug('RECV\n')

        while True:
            # receive data header
            header_name = data_name + '_header'
            try:
                header = self._recv(header_name, data_bytes=self.buffer_size, is_header=True)
            except timeout:
                raise

            self.__logger.debug('header: ' + str(header))

            # whether get the correct packet
            if data_name == header['data_name']:  # correct, break the while loop
                self.__logger.debug('matching header name')
                break

            else:  # incorrect, keep waiting
                self.__logger.error('"%s" Receive Wrong data. Expect to receive "%s" instead of "%s" !'
                                    % (self.type, data_name, header['data_name']))

        data = self._recv(data_name, data_bytes=header['data_bytes'])

        self.__logger.debug('"%s" Receive "%s" Successfully !' % (self.type, data_name))

        return data

    def is_right_conn(self, client_name):
        if self.is_server:
            self.awake()
            recv_client_name = self.recv('client_name')
            if recv_client_name == client_name:
                self.send(True, 'is_conn_or_not')
                self.__logger.debug('accept "%s" connection!' % recv_client_name)
                return True
            else:
                self.send(False, 'is_conn_or_not')
                self.__logger.debug('NOT accept "%s" connection !' % recv_client_name)
                return False
        else:
            self.sleep()
            self.send(client_name, 'client_name')

            try:
                conn = self.recv('is_conn_or_not')
            except timeout:
                conn = False

            if conn:
                self.__logger.debug(
                    '"%s" connect to "%s" Successfully!' % (client_name, self.server_host_port))
                return True
            else:
                self.__logger.debug('"%s" CANT''T connect to "%s" !' % (client_name, self.server_host_port))

                self.socket.close()
                return False

    def sleep(self):
        _ = self.recv('awake')

    def awake(self):
        self.send(True, 'awake')

    def accept(self):
        self.conn, self.addr = self.socket.accept()
        self.__logger.debug('accept %s connection !' % str(self.addr))

    def connect(self):
        self.socket.connect(self.server_host_port)
        self.__logger.debug('connect to %s !' % str(self.server_host_port))
        # self.socket.settimeout(30)


    def close(self):
        if self.is_server:
            self.conn.close()
        else:
            self.socket.close()



