import numpy as np

# import PEF_Calculator as PEF
import data_loading_mod as dlm
from zmqhelper import Client
import json
import time
import base64
import random
import zlib
import os
from os import listdir
from os.path import isfile, join
import analysishelper.coinclib as cl

# import bellhelper.redisHelper as rh
import redis
import yaml

# import shutil
# from PEF_Accumulator import PEF_Accumulator
# import PEF_Accu

# March Frequencies
# freq = np.array([[3551853,   81632,   85159,    2437],
#                [3623020,    8211,   56850,   30070],
#                [3625515,   55481,    8894,   28782],
#                [3667943,   13706,   12605,   24827]])

ip = "127.0.0.1"
port = "5553"

extractorZMQ = Client(ip, port)

freq = np.array(
    [
        [32266283, 600862, 580732, 18985],
        [32801781, 70119, 404229, 195257],
        [32803817, 422004, 61941, 197106],
        [33128142, 94035, 89267, 171053],
    ]
)


def compress_binary_data(data, aggregate=False):
    # # f = open(fname, 'a+')
    # '''
    # data types:
    # 'u1' = 1-byte unsinged integer
    # 'u8' = 8-byte unsigned integers
    # '''
    # sA = data['alice']['Setting'].astype('u1') # Alice Settings
    # sB = data['bob']['Setting'].astype('u1') # Bob settings
    # eA = data['alice']['Outcome'].astype('u8') # Alice outcome
    # eB = data['bob']['Outcome'].astype('u8') # Bob outcome
    # if aggregate:
    #     dataType = [('sA','u1'),('sB','u1'),('eA','u1'), ('eB','u1')]
    # else:
    #     dataType = [('sA','u1'),('sB','u1'),('eA','u8'), ('eB','u8')]

    # Create a structured array. Each row represents the results from one trial.
    # data = np.zeros(len(sA), dtype = dataType)

    # data['sA'] = sA
    # data['sB'] = sB
    # data['eA'] = eA
    # data['eB'] = eB

    # data.tofile(fname)
    if aggregate:
        dataType = [("SA", "u1"), ("SB", "u1"), ("OA", "u1"), ("OB", "u1")]
    else:
        dataType = [("SA", "u1"), ("SB", "u1"), ("OA", "u8"), ("OB", "u8")]
    data = data.astype(dtype=dataType)
    binData = data.tobytes()
    compressedData = zlib.compress(binData, level=-1)

    return compressedData


def convert_str_to_bytes(strData):
    data = base64.b64decode(strData)
    return data


def convert_bytes_to_str(binData):
    strData = base64.b64encode(binData).decode("utf-8")
    return strData


def send_message(con, cmd, params, timeout=10000):
    msg = {}
    msg["cmd"] = str(cmd).lower()
    msg["params"] = {}
    for key, value in params.items():
        # print(key, type(value))
        if type(value) == bytes:
            value = convert_bytes_to_str(value)
        elif type(value) == np.ndarray:
            value = value.tolist()
            # value = convert_str_to_bytes(value.tolist())
        msg["params"][key] = value
    msgJSON = json.dumps(msg)
    ret = con.send_message(msgJSON, timeout=timeout)
    ret = json.loads(ret)
    return ret


def load_freqs(path, nStart=1, nFiles=1, nPulses=11):
    # path = '/Users/lks/Documents/BellData/2022/processed/test/'
    # date = '2022_10_06/'
    # path = '/Users/lks/Documents/BellData/2023/2023_05_28/'
    # file_name_compressed = '2023_05_28_16_10_compressed_bafyriqayarc5bphqpeod2xr3of2wnpresrcudtebo5rog2u6skkjuvdybtdyu3lu4wec23ghhqjtftxysqyrhscmlo3eh7dwtshzl7p24g7dw_production_run_2_0_60s.dat'
    # date = ''

    # fname = path+file_name_compressed

    files = processFilesInDir(path)
    # print('FILES')
    # print(files)
    # print('')

    # dataFormat = [('SA','u1'),('SB','u1'),('OA','u8'),('OB','u8')]
    dataFormat = [("SA", "u1"), ("SB", "u1"), ("OA", "u1"), ("OB", "u1")]
    totalFreq = np.zeros((4, 4)).astype(int)
    pulses = np.array(range(1, nPulses + 1))
    # print('')
    # print('Number of pulses', len(pulses))
    nStop = nFiles + nStart
    rawData = None
    # if nStop>len(files):
    #     print('something')
    #     nStop = len(files)
    # print(range(nStart, nStop))
    # for i in range(nStart, nStop):
    if (nFiles > 0) & (nFiles < len(files)):
        files = [files[0]]
    # files = [files[0]]
    for i in range(len(files)):
        # print(i)
        fname = files[i]
        print(fname)
        data = dlm.read_data_file(path, fname, dataFormat=dataFormat)
        with open(fname, mode="rb") as f:
            compressedData = f.read()
        # freq = dlm.get_freqs(data)
        # if rawData is None:
        #     rawData = data
        # else:
        #     rawData = np.concatenate((rawData, extraData), axis=0)
        # totalFreq += freq.astype(int)
        freq, extraData = dlm.get_freqs_pulse_encoding(
            data, pulses, dataFormat=dataFormat
        )
        # print('Before \n', freq, '\n')
        # freq = fix_freqs(freq)
        freqCH = fix_freqs_ch(freq)
        CH, CHn, ratio, pValue = cl.calc_violation(freqCH)
        print("pValue:", pValue)
        if pValue < 1e-30:
            # print(CH, CHn, ratio, pValue)
            totalFreq += freq.astype(int)
        else:
            print("oops!")
        # if rawData is None:
        #     rawData = extraData
        # else:
        #     rawData = np.concatenate((rawData, extraData), axis=0)
    # rawData = rawData.astype(dataFormat)
    # binData = rawData.tobytes()
    # compressedData = zlib.compress(binData, level=-1)
    # compressedData = compress_binary_data(rawData, aggregate=True)
    print("")
    print("TOTAL Frequencies")
    print(totalFreq)
    print("")
    freqCH = fix_freqs_ch(totalFreq)
    CH, CHn, ratio, pValue = cl.calc_violation(freqCH)
    print("")
    print("pValue")

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
    cmd = "calc_pefs"
    params = {}
    # params['cmd'] = 'calc_pefs'
    params["freq"] = freq.tolist()
    params["beta"] = beta
    params["epsilonBias"] = epsilonBias
    params["isQuantum"] = isQuantum
    # msg = json.dumps(params)

    # result = extractorZMQ.send_message(msg, timeout=100000)
    result = send_message(extractorZMQ, cmd, params)
    # result = json.loads(result)
    pefs = np.array(result["pefs"])
    gain = float(result["gain"])
    # print('')
    # print("RESULTS of PEFS")
    # print(pefs, gain)
    return pefs, gain


def find_optimal_beta(
    freq, epsilonBias, delta, nBitsOut, error, fracSmoothness, isQuantum
):
    # @TODO check that all parameters are there.
    cmd = "find_beta"
    params = {}
    params["freq"] = freq
    params["epsilonBias"] = epsilonBias
    params["delta"] = delta
    params["nBitsOut"] = nBitsOut
    params["error"] = error
    params["fracSmoothness"] = fracSmoothness
    params["isQuantum"] = isQuantum

    res = send_message(extractorZMQ, cmd, params, timeout=100000)
    beta = res["beta"]
    # beta = extractorZMQ.send_message(msg, timeout=100000)
    beta = float(beta)
    return beta


def calculate_entropy(
    conn, expFreq, pefs, errorSmoothness, beta, epsilonBias, isQuantum=True
):
    cmd = "calc_entropy"
    params = {}
    params["freq"] = expFreq
    params["pefs"] = pefs
    params["errorSmoothness"] = errorSmoothness
    params["beta"] = beta
    params["epsilonBias"] = epsilonBias
    params["isQuantum"] = isQuantum

    result = send_message(conn, cmd, params)
    entropy = np.array(result["entropy"])
    # entropy = int(np.floor(entropy))
    return entropy


def calculate_extractor_properties(
    conn, nBitsOut, beta, gain, errorSmoothness, errorExtractor, isQuantum
):
    cmd = "calc_extractor_properties"
    params = {}
    params["nBitsOut"] = nBitsOut
    # params['nBitsIn'] = nBitsIn
    params["gain"] = gain
    params["errorSmoothness"] = errorSmoothness
    params["errorExtractor"] = errorExtractor
    params["beta"] = beta
    params["gain"] = gain
    params["epsilonBias"] = epsilonBias
    params["isQuantum"] = isQuantum

    result = send_message(conn, cmd, params)
    nBitsThreshold = int(result["nBitsThreshold"])
    nTrialsNeeded = int(result["nTrialsNeeded"])
    inputLenth = 2 * nTrialsNeeded
    seedLength = int(np.ceil(result["seedLength"]))

    return nBitsThreshold, nTrialsNeeded, inputLenth, seedLength


def process_entropy(conn, params):
    cmd = "process_entropy"
    # cmd = 'calc_entropy'

    result = send_message(conn, cmd, params)
    # print(result)
    entropy = np.array(result["entropy"])
    isThereEnoughEntropy = result["isThereEnoughEntropy"]
    # nBitsThreshold = int(result['nBitsThreshold'])
    nBitsThreshold = None
    return entropy, nBitsThreshold, isThereEnoughEntropy


def get_experiment_parameters(conn, params):
    cmd = "get_experiment_parameters"
    # print('params for jasper 1')
    # print(params)
    # print('')
    result = send_message(conn, cmd, params)
    print("")
    print("Params")
    print(result)
    print("")

    pefs = np.array(result["pefs"])
    beta = float(result["beta"])
    gain = float(result["gain"])
    nBitsThreshold = int(result["nBitsThreshold"])
    nTrialsNeeded = float(result["nTrialsNeeded"])
    seedLength = int(result["seedLength"])

    return pefs, beta, gain, nBitsThreshold, nTrialsNeeded, seedLength, result


def trevisan_extract(
    conn,
    data,
    seed,
    entropy,
    nBitsOut,
    errorExtractor,
    stoppingCriteria,
    isQuantum,
):
    cmd = "extract"
    params = {}
    params["data"] = data
    params["nBitsOut"] = nBitsOut
    params["errorExtractor"] = errorExtractor
    params["seed"] = seed
    params["entropy"] = entropy
    params["stoppingCriteria"] = stoppingCriteria
    params["isQuantum"] = isQuantum

    result = send_message(conn, cmd, params, timeout=100000)
    outBits = result["outBits"]
    return outBits


def read_data_file(
    path,
    file_name,
    dataFormat=[("SA", "u1"), ("SB", "u1"), ("OA", "u8"), ("OB", "u8")],
):
    with open(
        join(path, file_name), mode="rb"
    ) as fh:  # changing from 'r' to 'rb' for read binary file
        # data = fh.read()
        binData = zlib.decompress(fh.read())
        # print(binData)
        # data = bytes(binData, 'utf-8')
        # data = zlib.compress(data, level=-1)
        # zlib.compress(binData, level=-1)
        data = np.frombuffer(binData, dtype=dataFormat)
        # data = binData
    return data


# def read_data_file_return_raw(path, file_name):
#     with open(join(path, file_name), mode ='rb') as fh: #changing from 'r' to 'rb' for read binary file
#         data = fh.read()
#         # binData = zlib.decompress(fh.read())
#         # # data = bytes(binData, 'utf-8')
#         # # data = zlib.compress(data, level=-1)
#         # # zlib.compress(binData, level=-1)
#         # data = np.frombuffer(binData, dtype=dataFormat)
#         # # data = binData
#     return data


def read_data_buffer(
    data, dataFormat=[("SA", "u1"), ("SB", "u1"), ("OA", "u8"), ("OB", "u8")]
):
    binData = zlib.decompress(data)
    # print(binData)
    data = np.frombuffer(binData, dtype=dataFormat)
    return data


def processFilesInDir(path):
    # print(path)
    # print(os.listdir(path))
    files = [
        f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))
    ]
    files.sort()
    # print(files)
    # print('')
    files = [path + f for f in files if ".dat" in f]
    return files


def fix_freqs(freqs):
    freqs[:, 1] += freqs[:, 3]
    freqs[:, 2] += freqs[:, 3]
    # print('after \n', freqs)
    return freqs


def fix_freqs_ch(freq):
    freqCH = np.zeros((4, 4))
    freqCH[0, :] = freq[3, :]
    freqCH[1:3, :] = freq[1:3, :]
    freqCH[3, :] = freq[0, :]
    freqCH = fix_freqs(freqCH)
    return freqCH


def create_seed(seedLength):
    random_array = np.random.randint(2, size=seedLength)
    return random_array


def load_config_from_file(fname):
    config_fp = open(fname, "r")
    config = yaml.load(config_fp, Loader=yaml.SafeLoader)
    config_fp.close()
    return config


# def main():
#     path_to_data = '/Users/lks/Documents/BellData/2023/2023_05_28/'
#     file_name_compressed = '2023_05_28_16_10_compressed_bafyriqayarc5bphqpeod2xr3of2wnpresrcudtebo5rog2u6skkjuvdybtdyu3lu4wec23ghhqjtftxysqyrhscmlo3eh7dwtshzl7p24g7dw_production_run_2_0_60s.dat'

#     data = read_data_file(path_to_data, file_name_compressed)
#     print(data)


# if __name__ == '__main__':
#     main()
# # expFreq, rawData = load_freqs(nStart=1,nFiles=1)

# # expFreq = np.array([[3587826, 66849, 64304, 2000],
# #                 [3644178, 7733, 44673, 21698],
# #                 [3642347, 47090, 6879, 22013],
# #                 [3681573, 10585, 9913, 18796]])


def calc_pefs_test_data(path, nPulses, nFiles):
    isQuantum = False

    if isQuantum:
        delta = 4e-8
    else:
        delta = 0

    nBitsOut = 512
    error = 2 ** (-64)
    fracSmoothness = 0.8
    errorSmoothness = fracSmoothness * error
    errorExtractor = (1 - fracSmoothness) * error
    epsilonBias = 1e-3

    params = {}

    params["epsilonBias"] = epsilonBias
    params["nBitsOut"] = nBitsOut
    # # params['gain'] = gain
    params["errorSmoothness"] = errorSmoothness
    params["errorExtractor"] = errorExtractor
    params["isQuantum"] = isQuantum

    allFreq, rawData = load_freqs(
        path, nStart=0, nFiles=nFiles, nPulses=nPulses
    )
    print("FREQS", allFreq)
    print("")

    params["data"] = rawData
    params["freq"] = allFreq
    params["stoppingCriteria"] = int(15e6)
    print("stoppingCriteria", params["stoppingCriteria"])
    # params['nBitsThreshold'] = nBitsThreshold
    (
        pefs,
        beta,
        gain,
        nBitsThreshold,
        nTrialsNeeded,
        seedLength,
        result,
    ) = get_experiment_parameters(extractorZMQ, params)
    params["gain"] = gain
    params["pefs"] = pefs
    params["beta"] = beta
    params["nBitsThreshold"] = nBitsThreshold
    params["nTrialsNeeded"] = nTrialsNeeded
    params["seedLength"] = seedLength

    params["freq"] = allFreq
    entropy, nBitsThreshold, success = process_entropy(extractorZMQ, params)

    # print(entropy, nBitsThreshold,  success)

    BELL_REQUEST_PARAMS = {
        "epsilonBias": epsilonBias,
        "nBitsOut": nBitsOut,
        "errorSmoothness": errorSmoothness,
        "errorExtractor": errorExtractor,
        "isQuantum": params["isQuantum"],
        "nBitsThreshold": nBitsThreshold,
        "stoppingCriteria": 15000000,
    }
    seed = create_seed(int(np.ceil(seedLength)))
    print("length of seed is ", seedLength, "isQuantum", isQuantum)
    outbits = trevisan_extract(
        extractorZMQ,
        params["data"],
        seed,
        entropy,
        BELL_REQUEST_PARAMS["nBitsOut"],
        BELL_REQUEST_PARAMS["errorExtractor"],
        BELL_REQUEST_PARAMS["stoppingCriteria"],
        BELL_REQUEST_PARAMS["isQuantum"],
    )

    return gain, result


if __name__ == "__main__":
    gains = []
    nPulses = [14]
    results = []
    nFiles = -1

    # path = '/Users/lks/Documents/BellData/2023/2023_05_28/'
    # path = '/Users/lks/Documents/BellData/2023/2023_04_26/'

    path = "/Users/lks/Downloads_old/bell_data_test_errors/trevisan/"
    # fname_error = '2023_07_20_18_20_compressed_bafyriqdsc4t6cwcat5pgj5zwm5kstmidmlvudeea7erfk5yhqpgx4u7tauntngkwsgctlqav75zo6wuhizp4q62zx3qekrjyikfdm5ds2qnwg_production_run_3_6_60s.dat'
    fname_error_2 = "2023_08_10_10_40_compressed_bafyriqhcqo23aqrup2hiz5tv6cd66ciux3bgaxkpvbifcxjoxqnl5kninduecy53mkux4ep3nzjqhyjoxxyzyrr2cnfsmkwms4thsvkykv3sa_production_run_3_137_60s.dat"
    file_name_yaml = "2023_08_10_10_40_config_bafyriqhcqo23aqrup2hiz5tv6cd66ciux3bgaxkpvbifcxjoxqnl5kninduecy53mkux4ep3nzjqhyjoxxyzyrr2cnfsmkwms4thsvkykv3sa_production_run_3_137_60s.yaml"

    config = load_config_from_file(path + file_name_yaml)
    # print(config)
    nP = 1
    nFiles = 1
    gain, result = calc_pefs_test_data(path, nP, nFiles)

    # for nP in nPulses:

    #     print('number of pulses', nP, nPulses)
    #     gain, result = calc_pefs_test_data(path, nP, nFiles)
    #     # print(nP, gain)
    #     gains.append((nP, gain))
    #     results.append((nP, results))

    # REDIS_IP = 'bellamd2.campus.nist.gov'
    # REDIS_PORT = 6379
    # r = redis.Redis(host=REDIS_IP,
    #                     port=REDIS_PORT,
    #                     db='')
    # # all_keys = r.keys('*')
    # # print(all_keys)

    # key1 = 'bell-requests'
    # key2 = 'bell-status'

    # msg = r.get(key2)
    # print(msg)
