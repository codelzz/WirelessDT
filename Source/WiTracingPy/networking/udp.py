"""
Reference:
[1] EpicGames/UnrealEngine UdpSocketReceiver.h https://github.com/EpicGames/UnrealEngine/blob/release/Engine/Source/Runtime/Networking/Public/Common/UdpSocketReceiver.h
"""
import queue

from udpthread.runnable import Runnable

# convert endpoint to string format
def endpoint_to_string(endpoint):
    return f"{endpoint[0]}:{endpoint[1]}"

# inherit from udpthread runnable and add socket specified members
class UdpSocketRunnable(Runnable):
    max_buffer_size = 65507  # value based on Unreal source code UdpSocketReceiver.h [1].

# Socket sender handle data sending task
class UdpSocketSender(UdpSocketRunnable):
    send_queue = queue.Queue()
    callback = None
    socket = None

    def __init__(self, socket, callback, wait_time=0.001):
        super(UdpSocketSender, self).__init__(wait_time=wait_time)
        self.socket = socket
        self.callback = callback

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

# Socket receiver handle data receiving task
class UdpSocketReceiver(UdpSocketRunnable):
    callback = None
    socket = None

    def __init__(self, socket, callback, wait_time=0.001):
        super(UdpSocketReceiver, self).__init__(wait_time=wait_time)
        self.socket = socket
        self.callback = callback

    def do(self):
        # gather received data from buffer and pass data
        # into callback

        try:
            byte_data, address = self.socket.recvfrom(self.max_buffer_size)
            if self.callback is not None:
                self.callback(byte_data, address)
        except ConnectionError as error:
            self.print(f'Disconnected! Reason: {error}')

