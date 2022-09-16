import numpy as np
import zmqhelper as zmqh
from PEF_Accumulator import PEF_Accumulator
import json

class PefServer(zmqh.Server):
    def __init__(self, port, n_workers):
        super().__init__(port, n_workers)

    def handle(self, message):
        string_message = message.decode('utf-8')
        params = json.loads(string_message)
        freq = np.array(params['freq'])
        pefs = np.array(params['pefs'])
        pefa = PEF_Accumulator(params['beta'], params['epsilon_bias'], params['delta'])
        entropy = pefa.accumulate_entropy(freq, pefs)
        return str(entropy).encode('utf-8')


if __name__ == '__main__':
    pefs = PefServer(port='5552', n_workers=2)
