from pathlib import Path
import subprocess
import numpy as np
import os

class TrevisanExtractorRun:
    def __init__(self, input, seed, entropy=None, nbits=512, error_prob=1.147943701974890e-43, file_dir='/dev/shm'):
        self.input = input #list
        # The settings of the experiment are either 1 or 2. So, we subtract to get 0 or 1.
        # self.seed = np.array(seed)-1 #ends as numpy array, can start as list of ints
        self.seed = seed
        self.entropy = np.floor(entropy) #float
        # self.entropy = entropy
        self.file_dir = Path(file_dir) #Location of storage on the container
        self.shm = file_dir
        self.nbits = nbits #int
        self.error_prob = (error_prob**2)/512/2 # Convert to error/bit
        # print('Error per bit:', self.error_prob)
        self.input_file = 'input.txt'
        self.seed_file = 'seed.txt'
        self.log_file = 'log.txt'
        self.output_file = 'output.txt'

    # def set_entropy(self, entropy):
    #     # Used to set the entropy to the value specified
    #     self.entropy =  entropy
    #     return

    def read_output(self):
        output = ''
        with open(self.file_dir / self.output_file, 'r') as file:
            for line in file:
                output += line
        return output

    def write_input(self):
        # print('joining input')
        # print(self.input[0:100])
        # str_input = ''.join([str(i) for i in self.input])
        str_input = str(self.input).replace(', ', '').strip('[').strip(']')
        self.input_length = len(str_input)
        print('str length', self.input_length, str_input[0:100])
        with open(self.file_dir / self.input_file, 'w+') as file:
            file.write(str(str_input))

    def write_seed(self):
        # str_seed = ''.join([str(i) for i in self.seed])
        str_seed = str(self.seed).replace(', ', '').strip('[').strip(']')
        print('seed length', len(str_seed), str_seed[0:10])
        with open(self.file_dir / self.seed_file, 'w+') as file:
            file.write(str(str_seed))

    def remove_files(self):
        files = [self.input_file, self.seed_file, self.log_file, self.output_file]
        
        for f in files:
            fname = os.path.join(self.shm, f)
            if os.path.exists(fname):
                os.remove(fname)

    def execute_extractor(self):
        '''
        Runs the command for the trevisan extractor. -t 6 sets the threads to 6.
        This can be changed based on the computer used.
        '''
        command_str = '/trev/extractor'
        command_str += ' -i ' + str(self.file_dir / self.input_file)
        command_str += ' -q ' + str(self.file_dir / self.seed_file)
        command_str += ' -n ' + str(self.input_length)
        command_str += ' -m ' + str(self.nbits)
        command_str += ' -a ' + str(self.entropy)
        command_str += ' -b '
        command_str += ' -e ' + str(self.error_prob)
        command_str += ' -o ' + str(self.file_dir / self.output_file)
        command_str += ' -t 6'
        command_str += ' > ' + str(self.file_dir / self.log_file)
        subprocess.run(command_str, shell=True, check=True)
