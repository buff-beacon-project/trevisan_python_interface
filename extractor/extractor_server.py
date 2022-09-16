import numpy as np
import zmqhelper as zmqh
from extractor_object import TrevisanExtractorRun
import json

class ExtractorServer(zmqh.Server):
    def __init__(self, port, n_workers):
        super().__init__(port, n_workers)
        self.updating = False

    def handle(self, message):
        str_message = message.decode('utf-8')
        dm = json.loads(str_message)
        ter = TrevisanExtractorRun(dm['outcomes'], dm['seed'], dm['entropy'])
        ter.write_input()
        ter.write_seed()
        ter.execute_extractor()
        outbits = ter.read_output()
        return outbits.encode('utf-8')
    '''
    def handle(self, message):
        if message[0] == 111:
            #First code is 'o' for 'output_data'
            #Fill out output data, and write seed&input files for extractor
            self.updating = True
            output_data = np.frombuffer(message[1:])
            ter = TrevisanExtractorRun(output_data)
            ter.write_input()
            ter.write_seed()
            self.updating = False
            return b'Done'
        elif message[0] == 101:
            return b'G'
        else:
            return b'Wrong'
    '''

if __name__ == '__main__':
    pefs = ExtractorServer(port='5553', n_workers=2)
