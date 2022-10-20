import numpy as np
import PEF_Calculator as PEF
import data_loading_mod as dlm 
from zmqhelper import Client
import json
import time
import base64
import random
import zlib
# from PEF_Accumulator import PEF_Accumulator
# import PEF_Accu

# March Frequencies
# freq = np.array([[3551853,   81632,   85159,    2437],
#                [3623020,    8211,   56850,   30070],
#                [3625515,   55481,    8894,   28782],
#                [3667943,   13706,   12605,   24827]])

ip='127.0.1'
port='5553'

extractorZMQ = Client(ip, port)

freq = np.array([[32266283,600862,580732, 18985],
                [32801781 ,70119,404229,195257],
                [32803817, 422004, 61941, 197106],
                [33128142,  94035, 89267, 171053]])

def convert_str_to_bytes(strData):
    data = base64.b64decode(strData)
    return data

def convert_bytes_to_str(binData):
    strData = base64.b64encode(binData).decode('utf-8')
    return strData

def send_message(con, cmd, params, timeout=10000):
    msg = {}
    msg['cmd'] = str(cmd).lower()
    msg['params'] = {}
    for key, value in params.items():
        # print(key, type(value))
        if type(value)==bytes:
            value = convert_bytes_to_str(value)
        elif type(value)==np.ndarray:
            value = value.tolist()
        msg['params'][key] = value
    msgJSON = json.dumps(msg)
    ret = con.send_message(msgJSON, timeout=timeout)
    ret = json.loads(ret)
    return ret


def load_freqs(nStart=1,nFiles=1, nPulses=11):
    path = '/Users/lks/Documents/BellData/2022/processed/'
    date = '2022_10_06/'

    files = dlm.processFilesInDir(path+date)

    dataFormat = [('SA','u1'),('SB','u1'),('OA','u8'),('OB','u8')]
    # dataFormat = [('SA','u1'),('SB','u1'),('OA','u1'),('OB','u1')]
    totalFreq = np.zeros((4,4)).astype(int)
    pulses = np.array(range(1,nPulses+1))
    print('')
    print('Number of pulses', len(pulses))
    nStop = nFiles + nStart
    rawData = None 
    if nStop>len(files):
        nStop = len(files)
    for i in range(nStart, nStop):
        fname = files[i]
        print(fname)
        data = dlm.read_data_file(path, fname, dataFormat=dataFormat)
        freq, extraData = dlm.get_freqs_pulse_encoding(data, pulses)
        totalFreq += freq.astype(int)
        if rawData is None:
            rawData = extraData 
        else:
            rawData = np.concatenate((rawData, extraData), axis=0)
    # rawData = rawData.astype(dataFormat)
    binData = rawData.tobytes()
    compressedData = zlib.compress(binData, level=-1)

    return totalFreq, compressedData 

# def load_freqs():
#     path = '/Users/lks/Documents/BellData/2022/processed/'
#     date = '2022_10_06/'

#     files = dlm.processFilesInDir(path+date)

#     dataFormat = [('SA','u1'),('SB','u1'),('OA','u8'),('OB','u8')]
#     # dataFormat = [('SA','u1'),('SB','u1'),('OA','u1'),('OB','u1')]
#     # totalFreq = np.zeros((4,4)).astype(int)
#     # pulses = np.array(range(1,12))
#     # pValueTotal = 0
#     nFiles = 1
#     nStart = 11

#     totalFreq = np.zeros((4,4)).astype(int)
#     nStop = nFiles + nStart 
#     if nStop>len(files):
#         nStop = len(files)
#     for i in range(nStart, nStop):
#         fname = files[i]
#         print(fname)
#         data = dlm.read_data_file(path, fname, dataFormat=dataFormat)

#         cmd = 'freqs'
#         params = {}
#         # params['cmd'] = 'freqs'
#         # params['data'] = convert_bytes_to_str(data)
#         params['data'] = data

#         res = send_message(extractorZMQ, cmd, params)
#         freqs = res['freqs']
#         freqs = np.array(freqs).astype('int')
#         # print('From extractor server, freqs:')
#         # print(freqs)
#         # print('')
#         # print('totalFreq before')
#         # print(totalFreq)
#         # print('')
#         totalFreq += freqs
#         # totalFreq = freqs + np.zeros((4,4)).astype('int')
#         # print('totalFreq loop', i)
#         # print(totalFreq)
#         # print('')

#     print('totalFreq')
#     print(totalFreq)
#     print('')
#     return totalFreq, data 

def calc_PEFs(freq, beta, epsilonBias, isQuantum):
    cmd = 'calc_pefs'
    params = {}
    # params['cmd'] = 'calc_pefs'
    params['freq'] = freq.tolist()
    params['beta'] = beta 
    params['epsilonBias'] = epsilonBias
    params['isQuantum'] = isQuantum
    # msg = json.dumps(params)

    # result = extractorZMQ.send_message(msg, timeout=100000)
    result = send_message(extractorZMQ, cmd, params)
    # result = json.loads(result)
    pefs = np.array(result['pefs'])
    gain = float(result['gain'])
    # print('')
    # print("RESULTS of PEFS")
    # print(pefs, gain)
    return pefs, gain

def find_optimal_beta(freq, epsilonBias, delta, nBitsOut, error, fracSmoothness, isQuantum):
    #@TODO check that all parameters are there.
    cmd = 'find_beta'
    params = {}
    params['freq'] = freq
    params['epsilonBias'] = epsilonBias
    params['delta'] = delta
    params['nBitsOut'] = nBitsOut
    params['error'] = error
    params['fracSmoothness'] = fracSmoothness
    params['isQuantum'] = isQuantum

    res = send_message(extractorZMQ, cmd, params, timeout=100000)
    beta = res['beta']
    # beta = extractorZMQ.send_message(msg, timeout=100000)
    beta = float(beta)
    return beta

def calculate_entropy(conn, expFreq, pefs, errorSmoothness, 
        beta, epsilonBias, isQuantum=True):
    cmd = 'calc_entropy'
    params = {}
    params['freq'] = expFreq
    params['pefs'] = pefs 
    params['errorSmoothness'] = errorSmoothness
    params['beta'] = beta 
    params['epsilonBias'] = epsilonBias 
    params['isQuantum'] = isQuantum

    result = send_message(conn, cmd, params)
    entropy = np.array(result['entropy'])
    # entropy = int(np.floor(entropy))
    return entropy

def calculate_extractor_properties(conn, nBitsOut, beta, gain, errorSmoothness, errorExtractor, isQuantum):
    cmd = 'calc_extractor_properties'
    params = {}
    params['nBitsOut'] = nBitsOut
    # params['nBitsIn'] = nBitsIn 
    params['gain'] = gain 
    params['errorSmoothness'] = errorSmoothness
    params['errorExtractor'] = errorExtractor
    params['beta'] = beta 
    params['gain'] = gain 
    params['epsilonBias'] = epsilonBias 
    params['isQuantum'] = isQuantum

    result = send_message(conn, cmd, params)
    nBitsThreshold = int(result['nBitsThreshold'])
    nTrialsNeeded = int(result['nTrialsNeeded'])
    inputLenth = 2*nTrialsNeeded
    seedLength = int(np.ceil(result['seedLength']))

    return nBitsThreshold, nTrialsNeeded, inputLenth, seedLength

def process_entropy(conn, params):
    cmd = 'process_entropy'
    # cmd = 'calc_entropy'

    result = send_message(conn, cmd, params)
    print(result)
    entropy = np.array(result['entropy'])
    isThereEnoughEntropy = result['isThereEnoughEntropy']
    # nBitsThreshold = int(result['nBitsThreshold'])
    nBitsThreshold = None
    return entropy, nBitsThreshold, isThereEnoughEntropy

def get_experiment_parameters(conn, params):
    cmd = 'get_experiment_parameters'

    result = send_message(conn, cmd, params)

    pefs = np.array(result['pefs'])
    beta = float(result['beta'])
    gain = float(result['gain'])
    nBitsThreshold = int(result['nBitsThreshold'])
    nTrialsNeeded = float(result['nTrialsNeeded'])
    seedLength = int(result['seedLength']) 

    return pefs, beta, gain, nBitsThreshold, nTrialsNeeded, seedLength

def trevisan_extract(conn, data, seed, entropy, nBitsOut, errorExtractor, stoppingCriteria):
    cmd = 'extract'
    params = {}
    params['data'] = data
    params['nBitsOut'] = nBitsOut
    params['errorExtractor'] = errorExtractor
    params['seed'] = seed 
    params['entropy'] = entropy
    params['stoppingCriteria'] = stoppingCriteria 

    result = send_message(conn, cmd, params, timeout=100000)
    outBits = result['outBits']
    return outBits
    

# expFreq, rawData = load_freqs(nStart=1,nFiles=1)

# expFreq = np.array([[3587826, 66849, 64304, 2000], 
#                 [3644178, 7733, 44673, 21698], 
#                 [3642347, 47090, 6879, 22013], 
#                 [3681573, 10585, 9913, 18796]])

isQuantum = True 

if isQuantum:
    delta = 4E-8
else:
    delta = 0


nBitsOut = 512
error = 2**(-64)
fracSmoothness = 0.8
errorSmoothness = fracSmoothness*error
errorExtractor = (1-fracSmoothness)*error
epsilonBias = 1E-3 

# beta = find_optimal_beta(freq, epsilonBias, delta, nBitsOut, error, fracSmoothness, isQuantum)
# pefs, gain = calc_PEFs(freq, beta, epsilonBias, isQuantum)

params = {}
# params['data'] = rawData
# params['freq'] = expFreq
# params['pefs'] = pefs 
# params['beta'] = beta 
params['epsilonBias'] = epsilonBias 
params['nBitsOut'] = nBitsOut
# params['gain'] = gain 
params['errorSmoothness'] = errorSmoothness
params['errorExtractor'] = errorExtractor
params['isQuantum'] = isQuantum 
# params['error'] = error 
# params['fracSmoothness'] = fracSmoothness
# beta = find_optimal_beta(freq, epsilonBias, delta, nBitsOut, error, fracSmoothness, isQuantum)
# pefs, gain = calc_PEFs(freq, beta, epsilonBias, isQuantum)
# Compute PEFs and other parameters 
nPulses = 11
freq, data = load_freqs(nStart=1,nFiles=1, nPulses=nPulses)
params['freq'] = freq
pefs, beta, gain, nBitsThreshold, nTrialsNeeded, seedLength = get_experiment_parameters(extractorZMQ, params)
print('')
print('Success', gain, nBitsThreshold, nTrialsNeeded, seedLength)
print('')

params['gain'] = gain 
params['pefs'] = pefs
params['beta'] = beta

expFreq, rawData = load_freqs(nStart=10,nFiles=1, nPulses=11)
params['data'] = rawData 
params['freq'] = expFreq
params['stoppingCriteria'] = int(15E6)
print('stoppingCriteria', params['stoppingCriteria'])
params['nBitsThreshold'] = nBitsThreshold
entropy, nBitsThreshold, success = process_entropy(extractorZMQ, params)
print(entropy, success)

if success:
    seed=[]
    for i in range(seedLength):
        seed.append(random.randint(0,1))

    print('generated n seed bits:', len(seed), seedLength, seed[0:10])

    result = trevisan_extract(extractorZMQ, params['data'], seed, entropy, params['nBitsOut'], params['errorExtractor'], params['stoppingCriteria'])
    print('Trevisan results', result)

# pulses = range(11,14)
# gainArray = []
# betaArray = []
# entropyArray = [] 
# successArray = []
# for p in pulses:
#     freq, data = load_freqs(nStart=1,nFiles=10, nPulses=p)
#     beta = find_optimal_beta(freq, epsilonBias, delta, nBitsOut, error, fracSmoothness, isQuantum)
#     pefs, gain = calc_PEFs(freq, beta, epsilonBias, isQuantum)
#     gainArray.append(gain)
#     betaArray.append(beta)

#     expFreq, rawData = load_freqs(nStart=10,nFiles=1, nPulses=p)
#     print(expFreq)
#     params['data'] = rawData 
#     # params['freq'] = expFreq
#     params['gain'] = gain 
#     params['pefs'] = pefs
#     params['beta'] = beta
#     entropy, nBitsThreshold, success = process_entropy(extractorZMQ, params)
#     print(entropy)
#     entropyArray.append(entropy)
#     successArray.append(success)

# print('')
# print('pulses')
# print(list(pulses))
# # print('')
# # print('gain')
# # print(gainArray)
# # print('')
# print('Entropy')
# print(entropyArray)
# print('')
# print('Success')
# print(successArray)





# params = {}
# params['data'] = rawData
# # params['freq'] = expFreq
# params['pefs'] = pefs 
# params['beta'] = beta 
# params['epsilonBias'] = epsilonBias 
# params['nBitsOut'] = nBitsOut
# params['gain'] = gain 
# params['errorSmoothness'] = errorSmoothness
# params['errorExtractor'] = errorExtractor
# params['gain'] = gain 
# params['isQuantum'] = isQuantum

# entropy, nBitsThreshold, success = process_entropy(extractorZMQ, params)
# # entropy = calculate_entropy(extractorZMQ,expFreq, pefs, errorSmoothness, beta, epsilonBias, isQuantum=isQuantum)
 
# print('')
# print('entropy', entropy, nBitsThreshold, success)

# if success:
#     nBitsThreshold, nTrialsNeeded, inputLenth, seedLength = calculate_extractor_properties(extractorZMQ, 
#         nBitsOut, beta, gain, errorSmoothness, errorExtractor, isQuantum)

#     print('nBitsThreshold', nBitsThreshold, 'nTrialsNeeded', nTrialsNeeded, 'seedLength', seedLength, 'NumberTrials', nTrialsNeeded )
#     seed=[]
#     for i in range(seedLength):
#         seed.append(random.randint(0,1))
#     # seed = random.sample(range(0, 1), seedLength)
#     print('generated n seed bits:', len(seed), seedLength, seed[0:10])

#     result = trevisan_extract(extractorZMQ, rawData, seed, entropy, nBitsOut, errorExtractor, nTrialsNeeded)
#     print('Trevisan results', result)
