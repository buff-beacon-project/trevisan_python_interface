# -*- coding: utf-8 -*-
"""
Created on 7/9/21
Authors: Aliza & Joe
"""
import data_loading_mod as dlm
import zmq
import numpy as np
import json
import time
from sympy import isprime
from math import ceil, log2, e


def get_experiment_data(experiment_socket):
    '''
    Gets the latest streamed data from a client running experimental_placeholder.property
    In this case, the manager is the server and the data is the socket.
    '''
    data_format = [('SA','u1'),('SB','u1'),('OA','u8'),('OB','u8')]
    data_bytes = experiment_socket.recv()
    data = np.frombuffer(data_bytes, dtype=data_format)
    return data

def update_pefs(params, freq_buffer, pef_socket):
    '''
    Sends an update request to a server running the pef_server.py program.
    '''
    params['freq'] = sum(freq_buffer).tolist()
    params_string = json.dumps(params)
    pef_socket.send_string('calculate ' + params_string)
    reply = pef_socket.recv()
    print(reply)
    return

def get_pefs(pef_socket):
    '''
    Gets the latest PEFs from the server running the pef_server.py program.
    '''
    pef_socket.send_string('get')
    pefs = np.array(json.loads(pef_socket.recv().decode('utf-8')))
    return pefs

def get_entropy(entropy_socket, params, freq, pefs):
    '''
    params: dictionary of the form
    {'beta':0.01, 'epsilon_bias':0.001, 'delta':4e-8, 'freq':None, 'pefs':None}

    freq: numpy array of the form output by data_loading_mod.py's get_freqs()
    pefs: numpy array of the form output by pef_server.py

    Gets the entropy from the server running the entropy_server.py program
    Sends a string of a json representation of a dictionary containing
    beta, epsilon_bias, delta, frequencies, and the current PEFs.
    '''
    params['freq'] = freq.tolist()
    params['pefs'] = pefs.tolist()
    params_string = json.dumps(params)
    entropy_socket.send_string(params_string)
    entropy_string = entropy_socket.recv().decode('utf-8')
    return float(entropy_string)

def trevisan_extract(extractor_socket, output_data, entropy, epsilon_bias):
    outcomes = np.concatenate((output_data['OA'], output_data['OB']), axis=None).tolist()
    print(len(outcomes))
    seed = np.concatenate((output_data['SA'], output_data['SB']), axis=None).tolist()
    s = json.dumps({'outcomes':outcomes, 'seed':seed, 'entropy':entropy})
    extractor_socket.send_string(s)
    return extractor_socket.recv().decode('utf-8')

context = zmq.Context()

experiment_socket = context.socket(zmq.REP)
experiment_socket.bind("tcp://*:5550")

pef_socket = context.socket(zmq.REQ)
pef_socket.connect("tcp://localhost:5551")

entropy_socket = context.socket(zmq.REQ)
entropy_socket.connect("tcp://localhost:5552")

extractor_socket = context.socket(zmq.REQ)
extractor_socket.connect("tcp://172.17.0.2:5553")

params = {'beta':0.01, 'epsilon_bias':0.001, 'delta':4e-8, 'freq':None, 'pefs':None}

freq_buffer_size = 10
freq_buffer = [None for i in range(0,freq_buffer_size)]
data_recieved = 0 #How many raw datasets have been sent by the experiment
while True:
    '''
    Poll the experimental setup for output data. If data is recieved, start
    processing the data
    '''
    data = get_experiment_data(experiment_socket)
    t = time.time()
    data_recieved += 1
    freq, output_data = dlm.get_freqs(data)
    experiment_socket.send(b'done')
    freq_buffer[(data_recieved-1) % freq_buffer_size] = freq
    if data_recieved == 10:
        update_pefs(params, freq_buffer, pef_socket)
    elif data_recieved > 10: #calculate entropy on the 11th+ dataset
        pefs = get_pefs(pef_socket)
        entropy = get_entropy(entropy_socket, params, freq, pefs)
        result = trevisan_extract(extractor_socket, output_data, entropy, params['epsilon_bias'])
        print(result)
        print(str(time.time()-t) + ' seconds')
    '''
    Send over data, entropy & seed to the trevisan extractor
    '''
