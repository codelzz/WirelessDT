"""
Reference:
[1] EpicGames/UnrealEngine UdpSocketReceiver.h https://github.com/EpicGames/UnrealEngine/blob/release/Engine/Source/Runtime/Networking/Public/Common/UdpSocketReceiver.h
"""
import socket
import time
import queue

from thread.runnable import Runnable

class UdpSocketRunnable(Runnable):
    # value based on Unreal source code UdpSocketReceiver.h [1].
    max_buffer_size = 65507  

class UdpSocketSender(UdpSocketRunnable):
    """
    Socket sender handle data sending task
    """
    def __init__(self, socket, callback, wait_time=0.001):
        super(UdpSocketSender, self).__init__(wait_time=wait_time)
        self.socket = socket
        self.callback = callback
        self.send_queue = queue.Queue()

    def sendto(self, byte_data, address):
        self.send_queue.put((byte_data, address))

    def do(self):
        # Send data from queue by socket with given address. 
        # And execute the callback after sent.
        if not self.send_queue.empty():
            byte_data, address = self.send_queue.get()
            self.socket.sendto(byte_data, address)

            if self.callback is not None:
                self.callback(byte_data, address)

class UdpSocketReceiver(UdpSocketRunnable):
    """
    Socket sender handle data receiving task
    """
    def __init__(self, socket, callback, wait_time=0.001):
        super(UdpSocketReceiver, self).__init__(wait_time=wait_time)
        self.socket = socket
        self.callback = callback

    def do(self):
        try:
            byte_data, address = self.socket.recvfrom(self.max_buffer_size)
            if self.callback is not None:
                self.callback(byte_data, address)
        except ConnectionError as error:
            self.print(f'Disconnected! Reason: {error}')

class UdpSocketClient:

    def __init__(self, endpoint, on_data_sent, on_data_recv, wait_time=0.001):
        self.endpoint = endpoint
        self.on_data_sent = on_data_sent
        self.on_data_recv = on_data_recv
        self.socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.socket.bind(self.endpoint)
        self.receiver = UdpSocketReceiver(self.socket, self.on_data_recv, wait_time=wait_time)
        self.sender = UdpSocketSender(self.socket, self.on_data_sent, wait_time=wait_time)

    def start(self):
        if self.sender is not None:
            self.sender.start()

        if self.receiver is not None:
            self.receiver.start()

    def sendto(self, byte_data, address):
        self.sender.sendto(byte_data=byte_data, address=address)

