import zmqhelper as zmqh
import json as json
import traceback
import extractor_jobs as jobs

class ExtractorServer(zmqh.Server):
    def __init__(self, port, n_workers, aggregate=True):
        if aggregate:
            self.dataFormat = [('SA','u1'),('SB','u1'),('OA','u1'),('OB','u1')]
        else:
            self.dataFormat = [('SA','u1'),('SB','u1'),('OA','u8'),('OB','u8')]

        print(self.dataFormat)
        super().__init__(port, n_workers)

    def handle(self,msg):
        # Msgout is returned from motor command
        try:
            print('')
            print('message received')
            inputs = json.loads(msg)
            cmd = inputs['cmd']
            params = inputs['params']
            # print("Received request: %s" % cmd)
            # print('cmd', cmd[0])
            cmd = cmd.lower()
            print('Received command:', cmd)
            if cmd == "extract":
                outBits = jobs.run_extractor(params, self.dataFormat)
                res = {}
                res['outBits'] = outBits

            elif cmd == 'freqs':
                freqs = jobs.get_freqs(params, self.dataFormat)
                res = {}
                res['freqs'] = freqs
                # msgout = freqs

            elif cmd == 'calc_pefs':
                pefs, gain = jobs.calc_PEFs(params)
                res = {}
                res['pefs'] = pefs
                res['gain'] = gain

            elif cmd == 'find_beta':
                beta = jobs.find_optimal_beta(params)
                res = {}
                res['beta'] = beta

            elif cmd == 'calc_entropy':
                entropy = jobs.calc_entropy(params)
                res = {}
                res['entropy'] = entropy

            elif cmd == 'process_entropy':
                entropy, success = jobs.process_entropy(params, self.dataFormat)
                res = {}
                res['entropy'] = entropy
                # res['nBitsThreshold'] = nBitsThreshold
                res['isThereEnoughEntropy'] = success
                print('entropy', entropy, success)

            elif cmd == 'calc_extractor_properties':
                nBitsThreshold, nTrialsNeeded, seedLength = jobs.compute_extractor_properties(params)
                res = {}
                res['nBitsThreshold'] = nBitsThreshold
                res['nTrialsNeeded'] = nTrialsNeeded
                res['seedLength'] = seedLength

            elif cmd == 'get_experiment_parameters':
                pefs, beta, gain, nBitsThreshold, nTrialsNeeded, seedLength = jobs.get_experiment_parameters(params)
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
            traceback.print_exc()
            res = {}
            res['error'] = "Error: "+str(e)
            # raise e
        msgout = jobs.encode_message_to_JSON(res)
        msgout = msgout.encode('utf-8')

        return msgout

if __name__ == '__main__':
    print('Starting Extractor Server')
    pefs = ExtractorServer(port='5553', n_workers=1, aggregate=True)
