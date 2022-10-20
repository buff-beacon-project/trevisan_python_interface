# from PEF_Analysis import PEF_Analysis
import numpy as np
import os as os
from os import listdir
from os.path import isfile, join
# import shutil
import zlib

# from scipy.stats import binom
# from scipy.stats import mode

#path = r'E:\2018_04_13_all_files_processed\processed\Experimenting' #source

# information about data collection runs contained in path. This must be done manually.
#run = sorted([f for f in listdir(self.path) if (".bin" in f)])



def data_processing(path, num_files, starting_index):
    """ @author: Aliza Siddiqui
    Executes all the processing (MADE FOR CALIBRATION)
    num_files will determine how many binary files must be taken to create frequencies matrix:
        1.) Read in binary data file
        2.) Get frequencies matrix of specific binary data file
        3.) Add those S+O to the overall frequencies matrix
    :param starting_index - The starting index in the data drive to began reading in files for calibration
    :return freq- frequencies matrix for all 16 possible S+O combination for all "num_files" binary data files
    """
    freq = np.zeros((4,4))
    files = os.listdir(path)
    output_data = {}
    files_processed = 0 #used to make sure we take the exact number of calibration files needed (10)
    while starting_index >=0 and files_processed <= num_files:
        data = read_data_file(path, files[starting_index])
        # print("data file " , files[starting_index], " read in")
        freqDataFile, output_data = get_freqs(data, output_data)
        # print("freq from data file: ", freqDataFile)
        freq = np.add(freq, freqDataFile)
        # print("freq overall: ", freq)
        starting_index = starting_index - 1
        files_processed = files_processed + 1
    # print("freq from all data files: ", freq)
    return freq



# def read_data_file(path, file_name, dataFormat= [('SA','u1'),('SB','u1'),('OA','u8'),('OB','u8')]):
#     """ @original author: Mohammad Alhejji
#         @Last modified: 02/17/2021 by Aliza Siddiqui
#     Reads in data from a binary file
#         'SA' settings of Alice in the specific trial
#         'SB' settings of Bob
#         'OA' outcomes of Alice
#         'OB' outcomes of Bob
#         'u1' unsigned one byte integer (8 bits) (uint8)
#         'u8' unsigned eight byte integer (64 bits)
#         NOTE: "Unsigned" refers to the fact that there are no negative bitstrings in this scenario
#     :param file_name - Path of binary file that is being read
#     :dataFormat - How each experiment result is encoded in the file
#     :return data - list of data lines
#     """
#     # print("Reading " + file_name + "\n")
#     with open(join(path, file_name), mode ='rb') as fh: #changing from 'r' to 'rb' for read binary file
#         data = fh.read()
#         # binData = zlib.decompress(fh.read())
#         # data = np.frombuffer(binData, dtype=dataFormat)

#          #constructs an array from the data in the binary file; -1 means read complete file
#     return data
def read_data_file(path, file_name, dataFormat= [('SA','u1'),('SB','u1'),('OA','u8'),('OB','u8')]):
    with open(join(path, file_name), mode ='rb') as fh: #changing from 'r' to 'rb' for read binary file
        # data = fh.read()
        binData = zlib.decompress(fh.read())
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

def read_data_buffer(data, dataFormat= [('SA','u1'),('SB','u1'),('OA','u8'),('OB','u8')]):
    binData = zlib.decompress(data)
    data = np.frombuffer(binData, dtype=dataFormat)
    return data

def restructure_outcomes(data, mask):
    """ @author: Aliza Siddiqui
    Restructures the outcomes for both Alice and Bob to filter out the pulses we want to consider only
    as well as simplifies the data by processing multiple click events as just a click event

    :param mask- a binary array that will determine what pulses we are considering for the experiment
                 based on our definition of a trial
                i.e. Considering 2nd and 3rd pulse out of 6 pulses would be [0, 0, 1, 1, 0, 0 ]
    :return freq- frequencies matrix for all 16 possible S+O combination for all "num_files" binary data files
    """

    #Applying a mask to disregard certain pulses or "slots" in our definition of a trial
    #i.e. out definition of a trial is only disregarding the third or fifth slot/pulse

    #Turning the lists into arrays
    data['OA'] = np.array(data['OA'])
    data['OB'] = np.array(data['OB'])
    # print('max', np.max(data['OA']), mask)

    data['OA'] = (data['OA']& mask > 1).astype(int)
    data['OB'] = (data['OB']& mask > 1).astype(int)



    # #Making sure the entries of the array are not decimal values
    # data['OA'] = data['OA'].astype(int)
    # data['OB'] = data['OB'].astype(int)

    # #Applying Mask
    # # data['OA']= np.bitwise_and(data['OA'], mask)
    # # data['OB'] = np.bitwise_and(data['OB'], mask)

    # data['OA']= mask.astype(int) & data['OA']
    # data['OB'] = data['OB'] & mask.astype(int)

    # #Replacing multiple clicks happening to just "a click happened" = 1
    # #Simplifies the data
    # indexA = data['OA'] > 0
    # data['OA'][indexA] = 1
    # indexB = data['OB'] > 0
    # data['OB'][indexB] = 1

    return data

    # mask = (2**pcSlots).sum().astype(int)

    # for s in settings:
    #     out00 = ((data['OA'][s]& mask < 1) & (data['OB'][s]& mask < 1)).astype(int).sum()
    #     out01 = ((data['OA'][s]& mask < 1) & (data['OB'][s]& mask > 0)).astype(int).sum()
    #     out10 = ((data['OA'][s]& mask > 0) & (data['OB'][s]& mask < 1)).astype(int).sum()
    #     out11 = ((data['OA'][s]& mask > 0) & (data['OB'][s]& mask > 0)).astype(int).sum()
        # print(out11)



def determine_mask(pulses):
    """
    Takes the list of pulses you are considering, creates a binary format, converts binary number to decimal and returns
    the decimal number as the mask
        :param pulses - list of pulses you are considering in the experiment
        :return mask - the mask is the decimal number based on the pulses you are considering
    """
    mask = np.sum(2**np.array(pulses))
    # mask = 0
    # for i in pulses:
    #     mask = mask + (2**i)

    return mask


def get_freqs_pulse_encoding(data, pulses,dataFormat=[('SA','u1'),('SB','u1'),('OA','u8'),('OB','u8')]):
    """ @author: Mohammad Alhejji
        @Last modified: 07/21/2021 by Joe Cavanagh
    Takes data from binary file and outputs the frequencies matrix f(c|z): a matrix where each entry is the
    number of times you got a specific pair of outcomes given a specific pair of settings choices
        :param data - the array created from the data in the binary file
        :return freq - frequencies matrix for all 16 possible S+O combination
    """
    # output_data = {}
    # output_data['SA'] = np.array(data['SA'])
    # output_data['SB'] = np.array(data['SB'])
    # output_data['OA'] = np.array(data['OA'])
    # output_data['OB'] = np.array(data['OB'])

    output_data = np.array(data, dtype=dataFormat)


    #create mask based on pulses considered
    # pulses = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    #only considering these pulses (NOTE: python follows zero-based indexing. These are actually pulses 4-14)
    mask = determine_mask(pulses)
    output_data = restructure_outcomes(output_data, mask)
    # print('max after', np.max(output_data['SA']))


    #print("creating frequencies array")
    # Find which round(trials) have which setting choices {1, 2}
    sett_a1_b1 = (output_data['SA'] == 1) & (output_data['SB'] == 1)
    sett_a1_b2 = (output_data['SA'] == 1) & (output_data['SB'] == 2)
    sett_a2_b1 = (output_data['SA'] == 2) & (output_data['SB'] == 1)
    sett_a2_b2 = (output_data['SA'] == 2) & (output_data['SB'] == 2)
    sett = [sett_a1_b1, sett_a2_b1, sett_a1_b2, sett_a2_b2]
    #array of elements that satisfy each condition; first element in array is a list of
    #rounds that have SA=1 and SB = 1

    # Find which rounds have which outcomes {=0, >0}
    out_a0_b0 = (output_data['OA'] == 0) & (output_data['OB'] == 0)
    out_a0_b1 = (output_data['OA'] == 0) & (output_data['OB'] > 0)
    out_a1_b0 = (output_data['OA'] > 0) & (output_data['OB'] == 0)
    out_a1_b1 = (output_data['OA'] > 0) & (output_data['OB'] > 0)
    out = [out_a0_b0, out_a1_b0, out_a0_b1, out_a1_b1]
    #array of elements that satisfy each condition; first element in array is a list of elements that have
    #OA = 0 and OB = 0

    freqArray = []
    for s in sett:
        for o in out:
            freqArray.append(np.sum(o*s))
    freq = np.array(freqArray).reshape(4,4)
    print('output data type', type(output_data))
    # print("Frequencies Matrix:\n" + str(freq))
    return freq, output_data#, out

def get_freqs(data):
    """ @author: Mohammad Alhejji
        @Last modified: 07/21/2021 by Joe Cavanagh
    Takes data from binary file and outputs the frequencies matrix f(c|z): a matrix where each entry is the
    number of times you got a specific pair of outcomes given a specific pair of settings choices
        :param data - the array created from the data in the binary file
        :return freq - frequencies matrix for all 16 possible S+O combination
    """
    output_data = {}
    output_data['SA'] = np.array(data['SA'])
    output_data['SB'] = np.array(data['SB'])
    output_data['OA'] = np.array(data['OA'])
    output_data['OB'] = np.array(data['OB'])


    #create mask based on pulses considered
    # pulses = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    #only considering these pulses (NOTE: python follows zero-based indexing. These are actually pulses 4-14)
    # mask = determine_mask(pulses)
    # output_data = restructure_outcomes(output_data, mask)
    # print('max after', np.max(output_data['SA']))

    # Find which round(trials) have which setting choices {1, 2}
    sett_a1_b1 = (output_data['SA'] == 1) & (output_data['SB'] == 1)
    sett_a1_b2 = (output_data['SA'] == 1) & (output_data['SB'] == 2)
    sett_a2_b1 = (output_data['SA'] == 2) & (output_data['SB'] == 1)
    sett_a2_b2 = (output_data['SA'] == 2) & (output_data['SB'] == 2)
    sett = [sett_a1_b1, sett_a2_b1, sett_a1_b2, sett_a2_b2]
    #array of elements that satisfy each condition; first element in array is a list of
    #rounds that have SA=1 and SB = 1

    # Find which rounds have which outcomes {=0, >0}
    out_a0_b0 = (output_data['OA'] == 0) & (output_data['OB'] == 0)
    out_a0_b1 = (output_data['OA'] == 0) & (output_data['OB'] > 0)
    out_a1_b0 = (output_data['OA'] > 0) & (output_data['OB'] == 0)
    out_a1_b1 = (output_data['OA'] > 0) & (output_data['OB'] > 0)
    out = [out_a0_b0, out_a1_b0, out_a0_b1, out_a1_b1]

    freqArray = []
    for s in sett:
        for o in out:
            freqArray.append(np.sum(o*s))
    freq = np.array(freqArray).reshape(4,4)
    # print("Frequencies Matrix:\n" + str(freq))
    return freq

# def get_freqs(data, pcSlots):
#     # settings = find_settings(data)
#     # print(data['OA'][data['OA']>0])

#     sett_a1_b1 = (data['SA'] == 1) & (data['SB'] == 1)
#     sett_a1_b2 = (data['SA'] == 1) & (data['SB'] == 2)
#     sett_a2_b1 = (data['SA'] == 2) & (data['SB'] == 1)
#     sett_a2_b2 = (data['SA'] == 2) & (data['SB'] == 2)
#     settings = [sett_a1_b1, sett_a2_b1, sett_a1_b2, sett_a2_b2]
#     nTot = len(data)
#     freq = np.array(np.zeros(16))
#     offset = 0
#     mask = (2**pcSlots).sum().astype('u8')

#     for s in settings:
#         out00 = ((data['OA'][s]& mask < 1) & (data['OB'][s]& mask < 1)).astype(int).sum()
#         out01 = ((data['OA'][s]& mask < 1) & (data['OB'][s]& mask > 0)).astype(int).sum()
#         out10 = ((data['OA'][s]& mask > 0) & (data['OB'][s]& mask < 1)).astype(int).sum()
#         out11 = ((data['OA'][s]& mask > 0) & (data['OB'][s]& mask > 0)).astype(int).sum()
#         # print(out11)

#         freq[0 + offset] = out00
#         freq[0 + offset + 4] = out01
#         freq[0 + offset +8] = out10
#         freq[0 + offset+12] = out11
#         offset += 1

#     freq = freq.reshape((4,4)).astype(int)
#     freq = np.transpose(freq)
#     freq[:,[1,2]] = freq[:,[2,1]]
#     return freq, data, out11 

# def calc_violation(stats):
#     #  J = P(++|ab)        - P(+0|a'b)           - P(0+|ab')         - P(++|a'b')
#     stats[:,2] += stats[:,3]
#     stats[:,1] += stats[:,3]
#     stats[[1,2]] = stats[[2,1]] 

#     pab11 = stats[3,3]
#     papb11 = stats[1,3]
#     pabp11 = stats[2,3]
#     papbp11 = stats[0,3]
#     papb10 = stats[2,1]-pabp11
#     pabp01 = stats[1,2]-papb11
#     chSingles = (stats[1,2]+stats[3,2]+stats[2,1]+stats[3,1])*1./2.

#     chCoinc = pab11+papb11+pabp11-papbp11
#     # Jparts = [stats['coin'][1,1], -stats['alice'][2,1]+stats['coin'][2,1], -stats['bob'][1,2]+stats['coin'][1,2], -stats['coin'][2,2]]
#     # #print('Jparts: %r'%Jparts)
#     # J = stats['coin'][1,1] - stats['alice'][2,1] - stats['bob'][1,2] - stats['coin'][2,2]
#     # #print('J: %d'%sum(Jparts))

#     # CH_single = (stats['alice'][1,1]*1. + stats['alice'][2,1] + stats['bob'][1,1] + stats['bob'][1,2] )/2.
#     # CH_coin = stats['coin'][1,1] + stats['coin'][2,1] + stats['coin'][1,2] - stats['coin'][2,2]
#     CH = chCoinc - chSingles
#     CHn = chCoinc/chSingles
#     print("CH violation:", CH)
#     print("CH normalized:", CHn)

#     # Pab11 = stats['coin'][1,1]
#     # Papb10 = stats['alice'][2,1]-stats['coin'][2,1]
#     # Pabp01 = stats['bob'][1,2]-stats['coin'][1,2]
#     # Papbp11 = stats['coin'][2,2]

#     Ntot = pab11 + papb10 + pabp01 + papbp11*1.
#     ratio = pab11*1./Ntot

#     pValue = calc_pvalue(Ntot, ratio)

#     print("Ratio:", ratio, "pValue:", pValue)
#     return(CH, CHn, ratio, pValue)

# def calc_pvalue(N, prob):
#     return(binom.cdf(N*(1-prob), N, 0.5))

def processFilesInDir(path):
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    files.sort()
    files = [path+f for f in files if ".bin" in f]
    return files

def main():
    # path = 'testdata/'
    # path = '/Users/lks/code/bellhelpers/bellhelper/data'
    # fname = 'test.bin.zip'

    path = '/Users/lks/Documents/BellData/2022/processed/'
    date = '2022_10_06/'

    files = processFilesInDir(path+date)
    # fn = '2022_10_05_23_39_suboptimal_test_run_two_2_60s.bin.zip'
    # fname = path+date+fn 
    # starting_index = 1
    # num_files = 1
    dataFormat = [('SA','u1'),('SB','u1'),('OA','u8'),('OB','u8')]
    # dataFormat = [('SA','u1'),('SB','u1'),('OA','u1'),('OB','u1')]
    # data_processing(path, num_files, starting_index)
    totalFreq = np.zeros((4,4)).astype(int)
    pulses = np.array(range(1,12))
    pValueTotal = 0
    nFiles = 1
    nStart = 10
    nStop = nFiles + nStart 
    if nStop>len(files):
        nStop = len(files)
    for i in range(nStart, nStop):
        fname = files[i]
        print(fname)
        data = read_data_file(path, fname, dataFormat=dataFormat)
        # print(data)
    
        freq, output_data, out = get_freqs_pulse_encoding(data, pulses)
        totalFreq += freq.astype(int)
        # print(freq)
        CH, CHn, ratio, pValue = calc_violation(freq)
        pValueTotal += np.log(pValue)
        # print('')
    print(totalFreq.tolist())
    CH, CHn, ratio, pValue = calc_violation(totalFreq)
    print('Total P-Value (log):', pValueTotal, 'Total Pulses:', len(pulses))

if __name__ == '__main__':
    main()
