import zmq, json
import numpy as np
import zmqhelper as zmqh
from PEF_Calculator import PEF_Calculator
from threading import Thread

class PefServer(zmqh.Server):
    def __init__(self, port, n_workers):
        super().__init__(port, n_workers)
        self.pefs = None
        self.pef_flag = 'None' #Either 'None' 'Updating', or 'Done'

    def handle(self, message):
        print(message)
        string_message = message.decode('utf-8')
        if string_message.split(' ')[0].lower() == 'calculate':
            params_string = ' '.join(string_message.split(' ')[1:])
            t = Thread(target=self.update_pefs, daemon=True, args=(params_string,))
            t.start()
            self.pef_flag = 'Updating'
            return b'updating'
        elif string_message.split(' ')[0].lower() == 'get':
            if self.pef_flag == 'Done':
                return json.dumps(list(self.pefs)).encode('utf-8')
            else:
                return self.pef_flag.encode('utf-8')
        else:
            print('Error')
            return b'ERROR: No command specified'

    def update_pefs(self, params_string):
        params = json.loads(params_string)
        freq = np.array(params['freq'])
        print(freq)
        pef_calculator = PEF_Calculator(freq.T, params['beta'], params['epsilon_bias'], params['delta'])
        self.pefs, exp_gain = pef_calculator._calc_PEF_and_gain(params['beta'], params['epsilon_bias'])
        print(self.pefs)
        pef_calculator.compute_LR_stat_strength(freq.T)
        self.pef_flag = 'Done'
        return

if __name__ == '__main__':
    pefs = PefServer(port='5551', n_workers=2)
