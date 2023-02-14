import socket
import queue

from udpthread.runnable import Runnable

class TcpSocketRunnable(Runnable):
    max_buffer_size = 65507  # value based on Unreal source code UdpSocketReceiver.h [1].

# HOST = "192.168.31.110"  # Standard loopback interface address (localhost)
# PORT = 7777  # Port to listen on (non-privileged ports are > 1023)

# with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#     s.bind((HOST, PORT))
#     s.listen()
#     conn, addr = s.accept()
#     with conn:
#         print(f"Connected by {addr}")
#         while True:
#             data = conn.recv(65507)
#             if not data:
#                 break
#             print(data)
#             # conn.sendall(data)

class TcpSocketServer:
    endpoint = None

    def __init__(self, endpoint, on_data_recv, wait_time=0.01):
        self.endpoint = endpoint
        self.on_data_recv = on_data_recv
        self.socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
        self.socket.bind(self.endpoint)
        self.listener = TcpSocketListener(socket=self.socket, callback=self.on_data_recv, wait_time=wait_time)

    def start(self):
        if self.listener is not None:
            self.listener.start()

    def __repr__(self):
        return f"<{self.__class__.__name__} IP={self.endpoint[0]} Port={self.endpoint[1]}>"


 # Socket listener handle data receiving task
class TcpSocketListener(TcpSocketRunnable):
    callback = None
    socket = None
    count = 0
    connection = None
    address = None

    def __init__(self, socket, callback, wait_time=0):
        super(TcpSocketListener, self).__init__(wait_time=wait_time)
        self.socket = socket
        self.callback = callback

    def do(self):
        self.socket.listen()
        self.connection, self.address = self.socket.accept()
        with self.connection:
            print(f"Connected by {self.address}")
            while True:
                data = self.connection.recv(self.max_buffer_size)
                if not data:
                    break
                self.callback(data, self.address)
                
                # self.count+=1
                # print(f"RECV:{self.count}")
            # self.connection.sendall(data)
