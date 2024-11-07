import numpy as np
from extractor_class import TrevisanExtractorRun
import json as json
import PEF_Calculator as PEF
import data_loading_mod as dlm
import base64
import codecs
import re
import math

def parseSeed(base64Seed):
    decoded = base64.b64decode(base64Seed)
    str = "".join(["{:08b}".format(x) for x in decoded])
    list = [int(x) for x in str]
    return np.array(list).tolist()

def bitStringToBase64(str):
    as_hex = "%x" % int(re.sub('[^01]', '', str), 2)
    as_hex_pad = as_hex.zfill(math.ceil(len(as_hex)/2)*2)
    b = codecs.decode(as_hex_pad, 'hex')
    return base64.b64encode(b).decode('utf-8')

def get_extractor_per_bit_error(errorExtractor, nBitsOut, isQuantum):
    """
    Compute the extractor error per bit to use with the Trevisan Extractor.
    Depending on whether QEFs or PEFs are used, this value can change.
    """
    error_prob_per_bit = None
    if isQuantum:
        error_prob_per_bit = errorExtractor**2 / (2 * nBitsOut)
    else:
        error_prob_per_bit = errorExtractor / nBitsOut
    return error_prob_per_bit

def convert_str_to_bytes(strData):
    data = base64.b64decode(strData)
    return data

def convert_bytes_to_str(binData):
    strData = base64.b64encode(binData).decode('utf-8')
    return strData

def encode_message_to_JSON(result):
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

def get_delta(isQuantum):
    if isQuantum:
        delta = 4E-8
    else:
        delta = 0
    return delta

def get_freqs(params, dataFormat):
    binData = convert_str_to_bytes(params['data'])
    # binData = binData.tobytes()
    data = dlm.read_data_buffer(binData, dataFormat)
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

def find_optimal_beta(params):
    freq = np.array(params['freq'])
    epsilonBias = float(params['epsilonBias'])
    # delta = float(params['delta'])
    nBitsOut = int(params['nBitsOut'])
    # error = float(params['error'])
    errorSmoothness = float(params['errorSmoothness'])
    errorExtractor = float(params['errorExtractor'])
    # fracSmoothness = float(params['fracSmoothness'])
    isQuantum = bool(params['isQuantum'])
    delta = get_delta(isQuantum)

    beta = PEF.find_optimal_beta(freq, epsilonBias, delta, nBitsOut, errorSmoothness, errorExtractor, isQuantum)
    return beta

def calc_PEFs(params):
    freq = np.array(params['freq'])
    beta = float(params['beta'])
    epsilonBias = float(params['epsilonBias'])
    isQuantum = bool(params['isQuantum'])
    delta = get_delta(isQuantum)
    pefs, gain = PEF.calc_PEFs(freq, beta, epsilonBias, delta)
    print(pefs)
    print(gain)
    return pefs, gain

def calc_entropy(params):
    freq = np.array(params['freq'])
    pefs = np.array(params['pefs'])
    beta = float(params['beta'])
    epsilonBias = float(params['epsilonBias'])
    errorSmoothness = float(params['errorSmoothness'])
    isQuantum = bool(params['isQuantum'])
    delta = get_delta(isQuantum)

    entropy = PEF.calculate_entropy(freq, pefs, errorSmoothness,
            beta, epsilonBias, delta, isQuantum=isQuantum)
    return entropy

def compute_extractor_properties(params):
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

def process_entropy(params, dataFormat):
    # nBitsThreshold, nTrialsNeeded, seedLength = compute_extractor_properties(params)
    # params['nBitsThreshold'] = nBitsThreshold
    freq = get_freqs(params, dataFormat)
    params['freq'] = freq
    entropy = calc_entropy(params)

    nBitsThreshold = float(params['nBitsThreshold'])
    success = bool(entropy>nBitsThreshold)
    return entropy, success

def get_experiment_parameters(params):
    print('finding optimal_beta')
    beta = find_optimal_beta(params)
    params['beta'] = beta
    print('beta', beta)

    pefs, gain = calc_PEFs(params)
    params['pefs'] = pefs
    params['gain'] = gain
    print('pefs', pefs)
    print('gain', gain)

    nBitsThreshold, nTrialsNeeded, seedLength = compute_extractor_properties(params)
    return pefs, beta, gain, nBitsThreshold, nTrialsNeeded, seedLength

def run_extractor(params, dataFormat):
    binData = convert_str_to_bytes(params['data'])
    params['data'] = None
    data = dlm.read_data_buffer(binData, dataFormat)
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

    outcomesReordered = outcomesReordered.astype(int)
    outcomesReordered = outcomesReordered.tolist()
    # print('OUTCOMES', outcomesReordered[0:100])

    seed = parseSeed(params['seed'])
    entropy = params['entropy']
    nBitsOut = int(params['nBitsOut'])
    errorExtractor = float(params['errorExtractor'])
    isQuantum = bool(params["isQuantum"])
    error_prob_per_bit = get_extractor_per_bit_error(
        errorExtractor, nBitsOut, isQuantum
    )
    # extractorObject = TrevisanExtractorRun(outcomesReordered, seed, entropy, nBitsOut, errorExtractor)
    extractorObject = TrevisanExtractorRun(
        outcomesReordered,
        seed,
        entropy,
        nBitsOut,
        error_prob_per_bit=error_prob_per_bit,
    )
    # Write the input and seed
    print('')
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

    return bitStringToBase64(outBits)#.encode('utf-8')