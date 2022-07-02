from networking.udp_socket_client import UdpSocketClient


class UnrealUdpSocketClient(UdpSocketClient):
    def __init__(self, server_endpoint, client_endpoint, on_data_sent, on_data_recv):
        super(UnrealUdpSocketClient, self).__init__(endpoint=client_endpoint,
                                              on_data_sent=on_data_sent,
                                              on_data_recv=on_data_recv)
        self.client_endpoint = client_endpoint
        self.server_endpoint = server_endpoint

    def send(self, byte_data):
        self.sendto(byte_data=byte_data, address=self.server_endpoint)

    def start(self):
        super().start()
        self.print("Start")

    def print(self, payload):
        print(f"[{self.__class__.__name__}] {payload}")
