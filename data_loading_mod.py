from PEF_Analysis import PEF_Analysis
import numpy as np
import os as os
from os import listdir
from os.path import isfile, join
import shutil
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
        print("data file " , files[starting_index], " read in")
        freqDataFile, output_data = get_freqs(data, output_data)
        print("freq from data file: ", freqDataFile)
        freq = np.add(freq, freqDataFile)
        print("freq overall: ", freq)
        starting_index = starting_index - 1
        files_processed = files_processed + 1
    print("freq from all data files: ", freq)
    return freq



def read_data_file(path, file_name, dataFormat= [('SA','u1'),('SB','u1'),('OA','u8'),('OB','u8')]):
    """ @original author: Mohammad Alhejji
        @Last modified: 02/17/2021 by Aliza Siddiqui
    Reads in data from a binary file
        'SA' settings of Alice in the specific trial
        'SB' settings of Bob
        'OA' outcomes of Alice
        'OB' outcomes of Bob
        'u1' unsigned one byte integer (8 bits) (uint8)
        'u8' unsigned eight byte integer (64 bits)
        NOTE: "Unsigned" refers to the fact that there are no negative bitstrings in this scenario
    :param file_name - Path of binary file that is being read
    :dataFormat - How each experiment result is encoded in the file
    :return data - list of data lines
    """
    print("Reading " + file_name + "\n")
    with open(join(path, file_name), mode ='rb') as fh: #changing from 'r' to 'rb' for read binary file
         data = np.fromfile(fh, dtype=dataFormat, count=-1)
         #constructs an array from the data in the binary file; -1 means read complete file
    return data

def restructure_outcomes(output_data, mask):
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
    output_data['OA'] = np.array(output_data['OA'])
    output_data['OB'] = np.array(output_data['OB'])


    #Making sure the entries of the array are not decimal values
    output_data['OA'] = output_data['OA'].astype(int)
    output_data['OB'] = output_data['OB'].astype(int)

    #Applying Mask
    output_data['OA']= np.bitwise_and(output_data['OA'], mask)
    output_data['OB'] = np.bitwise_and(output_data['OB'], mask)

    #Replacing multiple clicks happening to just "a click happened" = 1
    #Simplifies the data
    indexA = output_data['OA'] > 0
    output_data['OA'][indexA] = 1
    indexB = output_data['OB'] > 0
    output_data['OB'][indexB] = 1



def determine_mask(pulses):
    """ @author: Aliza Siddiqui
    Takes the list of pulses you are considering, creates a binary format, converts binary number to decimal and returns
    the decimal number as the mask
        :param pulses - list of pulses you are considering in the experiment
        :return mask - the mask is the decimal number based on the pulses you are considering
    """
    mask = 0
    for i in pulses:
        mask = mask + (2**i)
    return mask


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
    pulses = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    #only considering these pulses (NOTE: python follows zero-based indexing. These are actually pulses 4-14)
    mask = determine_mask(pulses)
    restructure_outcomes(output_data, mask)


    #print("creating frequencies array")
    # Find which round(trials) have which setting choices {1, 2}
    sett_a1_b1 = np.where((output_data['SA'] == 1) & (output_data['SB'] == 1))
    sett_a1_b2 = np.where((output_data['SA'] == 1) & (output_data['SB'] == 2))
    sett_a2_b1 = np.where((output_data['SA'] == 2) & (output_data['SB'] == 1))
    sett_a2_b2 = np.where((output_data['SA'] == 2) & (output_data['SB'] == 2))
    sett = [sett_a1_b1, sett_a2_b1, sett_a1_b2, sett_a2_b2]
    #array of elements that satisfy each condition; first element in array is a list of
    #rounds that have SA=1 and SB = 1

    # Find which rounds have which outcomes {=0, >0}
    out_a0_b0 = np.where((output_data['OA'] == 0) & (output_data['OB'] == 0))
    out_a0_b1 = np.where((output_data['OA'] == 0) & (output_data['OB'] > 0))
    out_a1_b0 = np.where((output_data['OA'] > 0) & (output_data['OB'] == 0))
    out_a1_b1 = np.where((output_data['OA'] > 0) & (output_data['OB'] > 0))
    out = [out_a0_b0, out_a1_b0, out_a0_b1, out_a1_b1]
    #array of elements that satisfy each condition; first element in array is a list of elements that have
    #OA = 0 and OB = 0

    # To find f(c, z), all we need is to find the intersection
    freq =  np.array([np.size(np.intersect1d(i,j)) for i in sett for j in out]).reshape(4,4)
    print("Frequencies Matrix:\n" + str(freq))
    return freq, output_data


def main():
    path = '/home/joe/Desktop/bellrand-master/2018_07_26_settings4/'
    starting_index = 8
    num_files = 9
    data_processing(path, num_files, starting_index)

if __name__ == '__main__':
    main()
