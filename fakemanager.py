import numpy as np
import zmq
import json
import time

freq_buffer = np.array([[14828499, 20247, 21081, 39893],
[14700691, 150422, 16012, 45361],
[14685622, 16396, 165442, 44033],
[14506915, 191754, 205253, 3425]])

params = {'beta' : 0.01, 'epsilon_bias': 0.001, 'delta': 4e-8, 'freq': None}
params['freq'] = freq_buffer.tolist()
params_string = json.dumps(params)
print(params_string)

msg = "calculate " + params_string

context = zmq.Context()
pef_socket = context.socket(zmq.REQ)
pef_socket.connect("tcp://localhost:5551")

pef_socket.send_string(msg)
reply = pef_socket.recv()
print(reply)
for k in range(0,100):
    time.sleep(0.1)
    pef_socket.send_string('retrieve')
    reply = pef_socket.recv()
    print(reply.decode('utf-8'))
    if reply.decode('utf-8').lower() != 'updating':
        print(json.loads(reply))
        break
