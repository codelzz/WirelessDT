"""
Reference:
[1] EpicGames/UnrealEngine UdpSocketReceiver.h https://github.com/EpicGames/UnrealEngine/blob/release/Engine/Source/Runtime/Networking/Public/Common/UdpSocketReceiver.h
"""
import queue

from thread.runnable import Runnable


def endpoint_to_string(endpoint):
    return f"{endpoint[0]}:{endpoint[1]}"


class UdpRunnable(Runnable):
    max_buffer_size = 65507     # value based on Unreal source code UdpSocketReceiver.h [1].


class UdpSender(UdpRunnable):
    send_queue = queue.Queue()
    callback = None
    socket = None

    def __init__(self, socket, callback, wait_time=0.001):
        super(UdpSender, self).__init__(wait_time=wait_time)
        self.socket = socket
        self.callback = callback

    def sendto(self, byte_data, address):
        self.send_queue.put((byte_data, address))

    def do(self):
        if not self.send_queue.empty():
            byte_data, address = self.send_queue.get()
            self.socket.sendto(byte_data, address)

            if self.callback is not None:
                self.callback(byte_data, address)


class UdpReceiver(UdpRunnable):
    callback = None
    socket = None

    def __init__(self, socket, callback, wait_time=0.001):
        super(UdpReceiver, self).__init__(wait_time=wait_time)
        self.socket = socket
        self.callback = callback

    def do(self):
        try:
            byte_data, address = self.socket.recvfrom(self.max_buffer_size)
            if self.callback is not None:
                self.callback(byte_data, address)
        except ConnectionError as error:
            self.print(f'Disconnected. Reason {error}')

