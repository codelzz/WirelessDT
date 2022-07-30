APP_CONFIG = {
	"max_buffer_size": 65507,
    "server_endpoint": ("127.0.0.1", 8800),
    "client_endpoint": ("127.0.0.1", 8000),
}

DATA_CONFIG = {
	# csv
	'motion':'data/motion.csv',
	'signal':'data/signal.csv',
	# head ['frame', 'x', 'y', 'z', 'velocity x', 'velocity y','velocity z', 'acceleration x', 'acceleration y', 'acceleration z']
	'motion_fields':['timestamp', 'x','y','z','vx','vy','vz','ax','ay','az'],
	'signal_fields':['timestamp', 'address', 'rssi'],
}

HARDWARE_CONFIG = {
	'ble_port':'COM7',
	'baudrate':96000,
}

# TX MAC Address:
# 75:ac:a7:b3:2e:45 <-
# 50:d0:9a:e3:43:6a