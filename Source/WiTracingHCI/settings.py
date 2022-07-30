APP_CONFIG = {
	# "max_buffer_size": 65507,
	"server_endpoint": ("192.168.31.110", 8800),
    # "server_endpoint": ("127.0.0.1", 8800),
    "client_endpoint": ("", 8000), # reach able by all address
}

DATA_CONFIG = {
	# csv
	'motion':'data/motion.csv',
	'signal':'data/signal.csv',
	# head ['frame', 'x', 'y', 'z', 'velocity x', 'velocity y','velocity z', 'acceleration x', 'acceleration y', 'acceleration z']
	'motion_fields':['timestamp', 'x','y','z','vx','vy','vz','ax','ay','az','pitch','yaw','roll'],
	'signal_fields':['timestamp', 'address', 'rssi'],
}

HARDWARE_CONFIG = {
	'ble_port':'COM7',
	'baudrate':96000,
}

# TX MAC Address:
# 75:ac:a7:b3:2e:45 <-
# 50:d0:9a:e3:43:6a

# t265
# x = -z
# y = x
# z = y
