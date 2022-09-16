import numpy as np
import zmq
import json
import time
import data_loading_mod as dlm


context = zmq.Context()
trex_socket = context.socket(zmq.REQ)
trex_socket.connect("tcp://172.17.0.2:5553")


path = '/home/joe/Desktop/apr2018/yanbaosdataset'
testfile = '2018_04_13_10_17_extracted_settings4_15_60s.bin'
testdata = dlm.read_data_file(path, testfile)
freq, output_data = dlm.get_freqs(testdata)
entropy = 3123.0325239871604

print(np.array(output_data))
outcomes = np.concatenate((output_data['OA'], output_data['OB']), axis=None).tolist()
seed = np.concatenate((output_data['SA'], output_data['SB']), axis=None).tolist()
print('----------------------------------------------------------')
#print(outputs)
print('----------------------------------------------------------')
#print(seed)
s = json.dumps({'outcomes':outcomes, 'seed':seed, 'entropy':entropy})
trex_socket.send_string(s)
print(trex_socket.recv().decode('utf-8'))
