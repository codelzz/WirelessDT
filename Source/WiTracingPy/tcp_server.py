import socket
import json
import csv
import pandas

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 7777  # Port to listen on (non-privileged ports are > 1023)
DATA_LIMIT = 216 / 2
END_LIMIT = 216 / 2

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print(f"Connected by {addr}")
        data_count = 0
        data_sum = []
        file_count = 0
        while True:
            data_count += 1
            data = conn.recv(65507)
            if not data:
                break
            if data_count % DATA_LIMIT == 0:
                print(f"Saving results to file {file_count} ...")
                pd = pandas.DataFrame(data_sum)
                pd.to_csv("out"+str(file_count)+".csv")
                data_sum = []
                file_count += 1
            if data_count == END_LIMIT:
                break
            # print(data)
            my_json = data.decode('utf8').replace("'", '"')
            my_json_list = my_json.split("\r\n}")[:-1]
            line = []
            for idx, tx_data in enumerate(my_json_list):

                tx_data += "\r\n}"
                try:
                    json_data = json.loads(tx_data)
                    if idx == 0:
                        rxx = json_data["rxx"]
                        rxy = json_data["rxy"]
                        rxz = json_data["rxz"]
                        rssi = json_data["rssi"]
                        line.append(rxx)
                        line.append(rxy)
                        line.append(rxz)
                        line.append(rssi)
                    else:
                        rssi = json_data["rssi"]
                        line.append(rssi)
                except Exception:
                    pass
            if line:
                data_sum.append(line)
