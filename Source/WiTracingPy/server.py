from networking.udp_socket_server import UdpSocketServer 
import settings
import time

if __name__ == "__main__":
    server = UdpSocketServer(settings.NETWORK_CONFIG['server_endpoint'])
    server.start()

    while True:
        time.sleep(60)
