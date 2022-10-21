import numpy as np
import zmqhelper as zmqh
from extractor_class import TrevisanExtractorRun
import json as json
import time
import PEF_Calculator as PEF
import data_loading_mod as dlm 
import base64

class ExtractorServer(zmqh.Server):
    def __init__(self, port, n_workers, aggregate=True):
        if aggregate:
            self.dataFormat = [('SA','u1'),('SB','u1'),('OA','u1'),('OB','u1')]
        else:
            self.dataFormat = [('SA','u1'),('SB','u1'),('OA','u8'),('OB','u8')]

        print(self.dataFormat)
        super().__init__(port, n_workers)

    # def handle(self, msg):
    #     '''
    #     To save time, one could split this up.
    #     Just send over the outcomes and seed first, then run
    #     extractor_object.write_input() and extractor_object.write_seed().
    #     Once entropy is send, use extractor_object.get_entropy(entropy) to
    #     update the entropy and run with extractor_object.execute_extractor()

    #     Inputs: A dictionary serialized with json containing the experimental
    #     outcomes, the seed (experiment settings), and the entropy.

    #     Output: a string of '1's and '0's encoded into a python bytes object.
    #     '''
    #     # Turn the bytes recieved into a string
        

    def run_extractor(self,params):
        print('')
        binData = self.convert_str_to_bytes(params['data'])
        params['data'] = None
        data = dlm.read_data_buffer(binData, self.dataFormat)
        binData = None

        outcomesReordered = np.array([[data['OA']],[data['OB']]])
        data = None
        outcomesReordered = outcomesReordered.transpose().flatten()
        # print('step 1')
        # print('')

        nTrials = int(params['stoppingCriteria'])
        if nTrials>-1:
            nBits = 2*nTrials
            # print('nBits', nBits, len(outcomesReordered))
            if len(outcomesReordered)>nBits:
                print('data too long, truncate to stoppingCriteria')
                outcomesReordered = outcomesReordered[0:nBits]
            else:
                # Need to pad out the results
                print('data too short, pad to stoppingCriteria')
                outcomesPadded = np.zeros(nBits)
                outcomesPadded[0:len(outcomesReordered)] = outcomesReordered
                outcomesReordered = outcomesPadded

        # print('step 2')
        outcomesReordered = outcomesReordered.astype(int)

        # print('get rid of vals >1')
        # outcomes[outcomes>0] = 1
        # print('before to list', outcomesReordered[0:100])
        # print('to list')
        outcomesReordered = outcomesReordered.tolist()
        print('OUTCOMES', outcomesReordered[0:100])

        seed = np.array(params['seed']).tolist()
        entropy = params['entropy']
        nBitsOut = int(params['nBitsOut'])
        errorExtractor = float(params['errorExtractor'])

        # extractorObject = TrevisanExtractorRun(params['outcomesReordered'], params['seed'], params['entropy'], params['nbits'], params['error_prob'])
        extractorObject = TrevisanExtractorRun(outcomesReordered, seed, entropy, nBitsOut, errorExtractor)
        # Write the input and seed
        print('extractor object created')
        extractorObject.write_input()
        print('write input')
        extractorObject.write_seed()
        print('write seed')
        extractorObject.execute_extractor()
        print('reading output')
        outBits = extractorObject.read_output()
        print('output bits', outBits)
        print('cleaning up')
        extractorObject.remove_files() 
        extractorObject = None
        print('files deleted, ready for more input')
        print('')

        return outBits#.encode('utf-8')

    def get_delta(self, isQuantum):
        if isQuantum:
            delta = 4E-8
        else:
            delta = 0
        return delta

    def get_freqs(self, params):
        binData = self.convert_str_to_bytes(params['data'])
        # binData = binData.tobytes()
        data = dlm.read_data_buffer(binData, self.dataFormat)
        # Truncate data to the stopping criteria
        # if params['stoppingCriteria']:
        if 'stoppingCriteria' in params:
            stoppingCriteria = int(params['stoppingCriteria'])
        else:
            stoppingCriteria = -1 
        data = data[0:stoppingCriteria]
        freq = dlm.get_freqs(data)
        freq = freq
        # print(freq, type(freq))
        return freq

    def extract_data(self, data):
        pass 

    def find_optimal_beta(self, params):
        freq = np.array(params['freq'])
        epsilonBias = float(params['epsilonBias'])
        # delta = float(params['delta'])
        nBitsOut = int(params['nBitsOut'])
        # error = float(params['error'])
        errorSmoothness = params['errorSmoothness']
        errorExtractor = params['errorExtractor']
        # fracSmoothness = float(params['fracSmoothness'])
        isQuantum = bool(params['isQuantum'])
        delta = self.get_delta(isQuantum)

        beta = PEF.find_optimal_beta(freq, epsilonBias, delta, nBitsOut, errorSmoothness, errorExtractor, isQuantum)
        return beta

    def calc_PEFs(self, params):
        freq = np.array(params['freq'])
        beta = float(params['beta'])
        epsilonBias = float(params['epsilonBias'])
        isQuantum = bool(params['isQuantum'])
        delta = self.get_delta(isQuantum)
        pefs, gain = PEF.calc_PEFs(freq, beta, epsilonBias, delta)
        print(pefs)
        print(gain)
        return pefs, gain

    def calc_entropy(self, params):
        freq = np.array(params['freq'])
        pefs = np.array(params['pefs'])
        beta = float(params['beta'])
        epsilonBias = float(params['epsilonBias'])
        errorSmoothness = float(params['errorSmoothness'])
        isQuantum = bool(params['isQuantum'])
        delta = self.get_delta(isQuantum)

        entropy = PEF.calculate_entropy(freq, pefs, errorSmoothness, 
                beta, epsilonBias, delta, isQuantum=isQuantum)
        return entropy

    def compute_extractor_properties(self, params):
        nBitsOut = int(params['nBitsOut'])
        gain  = float(params['gain'])
        errorSmoothness = float(params['errorSmoothness'])
        errorExtractor = float(params['errorExtractor'])
        beta  = float(params['beta'])
        gain  = float(params['gain'])
        epsilonBias  = float(params['epsilonBias'])
        isQuantum = bool(params['isQuantum'])

        nBitsThreshold = PEF.calc_threshold_bits(nBitsOut, errorExtractor, isQuantum=isQuantum)
        nTrialsNeeded = PEF.compute_minimum_trials(nBitsOut, beta, gain, errorSmoothness, isQuantum=isQuantum)
        nBitsIn = 2*nTrialsNeeded
        seedLength = PEF.calc_seed_length(nBitsOut, nBitsIn, errorExtractor, isQuantum=isQuantum)

        return nBitsThreshold, nTrialsNeeded, seedLength

    def process_entropy(self, params):
        # nBitsThreshold, nTrialsNeeded, seedLength = self.compute_extractor_properties(params)
        # params['nBitsThreshold'] = nBitsThreshold 
        freq = self.get_freqs(params) 
        params['freq'] = freq
        entropy = self.calc_entropy(params)

        nBitsThreshold = float(params['nBitsThreshold'])
        success = bool(entropy>nBitsThreshold)
        return entropy, success

    def get_experiment_parameters(self, params):
        # params['stoppingCriteria'] = -1
        # freq = self.get_freqs(params) 
        # params['freq'] = freq
        print('finding optimal_beta')
        beta = self.find_optimal_beta(params)
        params['beta'] = beta
        print('beta', beta)

        pefs, gain = self.calc_PEFs(params)
        params['pefs'] = pefs 
        params['gain'] = gain
        print('pefs', pefs)
        print('gain', gain)

        nBitsThreshold, nTrialsNeeded, seedLength = self.compute_extractor_properties(params)
        return pefs, beta, gain, nBitsThreshold, nTrialsNeeded, seedLength

    def handle(self,msg):
        print('')
        print('message received')
        inputs = json.loads(msg)
        cmd = inputs['cmd']
        params = inputs['params']
        # print("Received request: %s" % cmd)
        # print('cmd', cmd[0])
        cmd = cmd.lower()
        print('Received command:', cmd)

        # Msgout is returned from motor command
        try:
            if cmd == "extract":
                outBits = self.run_extractor(params)
                res = {}
                res['outBits'] = outBits

            elif cmd == 'freqs':
                freqs = self.get_freqs(params)
                res = {}
                res['freqs'] = freqs
                # msgout = freqs
                
            elif cmd == 'calc_pefs':
                pefs, gain = self.calc_PEFs(params)
                res = {}
                res['pefs'] = pefs
                res['gain'] = gain

            elif cmd == 'find_beta':
                beta = self.find_optimal_beta(params)
                res = {}
                res['beta'] = beta
                
            elif cmd == 'calc_entropy':
                entropy = self.calc_entropy(params)
                res = {}
                res['entropy'] = entropy
                
            elif cmd == 'process_entropy':
                entropy, success = self.process_entropy(params)
                res = {}
                res['entropy'] = entropy
                # res['nBitsThreshold'] = nBitsThreshold
                res['isThereEnoughEntropy'] = success
                print('entropy', entropy, success)

            elif cmd == 'calc_extractor_properties':
                nBitsThreshold, nTrialsNeeded, seedLength = self.compute_extractor_properties(params)
                res = {}
                res['nBitsThreshold'] = nBitsThreshold
                res['nTrialsNeeded'] = nTrialsNeeded
                res['seedLength'] = seedLength
                
            elif cmd == 'get_experiment_parameters':
                pefs, beta, gain, nBitsThreshold, nTrialsNeeded, seedLength = self.get_experiment_parameters(params)
                res = {}
                res['pefs'] = pefs
                res['beta'] = beta
                res['gain'] = gain
                res['nBitsThreshold'] = nBitsThreshold
                res['nTrialsNeeded'] = nTrialsNeeded
                res['seedLength'] = seedLength

            else:
                res = {}
                res['error'] = "Invalid Command"

        # Catch errors and return them
        except Exception as e:
            print("Error: %r" % e)
            res = {}
            res['error'] = "Error: "+str(e)
            raise e
        msgout = self.encode_message_to_JSON(res)
        msgout = msgout.encode('utf-8')

        return msgout

    def convert_str_to_bytes(self, strData):
        data = base64.b64decode(strData)
        return data

    def convert_bytes_to_str(self, binData):
        strData = base64.b64encode(binData).decode('utf-8')
        return strData

    def encode_message_to_JSON(self, result):
        for key, value in result.items():
            if type(value)==bytes:
                value = convert_bytes_to_str(value)
            if type(value)==np.ndarray:
                value = value.tolist()
            # if isinstance(value, bool):
            #     return str(value).lower()
            result[key] = value
        msgJSON = json.dumps(result)
        return msgJSON


if __name__ == '__main__':
    print('Starting Extractor Server')
    pefs = ExtractorServer(port='5553', n_workers=1, aggregate=True)
