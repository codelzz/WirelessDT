from bluetooth.ble import DiscoveryService

if __name__ == "__main__":
     service = DiscoveryService()
     devices = service.discover(2)

     for address, name in devices.items():
         print("name: {}, address: {}".format(name, address))