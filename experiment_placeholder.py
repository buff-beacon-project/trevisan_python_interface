import data_loading_mod as dlm
import numpy as np
import zmq

path = '/home/joe/Desktop/apr2018/yanbaosdataset'

testfiles = ['2018_04_13_10_13_extracted_settings4_11_60s.bin',
                '2018_04_13_10_14_extracted_settings4_12_60s.bin',
                '2018_04_13_10_15_extracted_settings4_13_60s.bin',
                '2018_04_13_10_16_extracted_settings4_14_60s.bin',
                '2018_04_13_10_23_extracted_settings4_21_60s.bin',
                '2018_04_13_10_18_extracted_settings4_16_60s.bin',
                '2018_04_13_10_19_extracted_settings4_17_60s.bin',
                '2018_04_13_10_20_extracted_settings4_18_60s.bin',
                '2018_04_13_10_21_extracted_settings4_19_60s.bin',
                '2018_04_13_10_22_extracted_settings4_20_60s.bin',
                '2018_04_13_10_17_extracted_settings4_15_60s.bin']



context = zmq.Context()
socket = context.socket(zmq.REQ)
print("Connecting to server…")
#socket.connect("tcp://172.17.0.2:5559")
socket.connect("tcp://localhost:5550")

for k in range(0, len(testfiles)):
    print('Reading data file ' + str(k))
    testdata = dlm.read_data_file(path, testfiles[k])
    print('Encoding data file ' + str(k))
    encoded = testdata.tobytes()
    print('Sending encoded data ' + str(k))
    socket.send(encoded)
    feedback = socket.recv()
    print("feedback: %s" % feedback)
