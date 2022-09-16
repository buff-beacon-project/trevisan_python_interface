from pathlib import Path
import subprocess
import numpy as np

class TrevisanExtractorRun:
    def __init__(self, input, seed, entropy, nbits=512, error_prob=1.147943701974890e-43, file_dir='/dev/shm'):
        self.input = input
        self.seed = np.array(seed)-1
        self.entropy = entropy
        self.file_dir = Path(file_dir) #Location of storage on the container
        self.nbits = nbits
        self.error_prob = error_prob
        self.input_file = 'input.txt'
        self.seed_file = 'seed.txt'
        self.log_file = 'log.txt'
        self.output_file = 'output.txt'

    def get_entropy(self, entropy):
        self.entropy =  entropy
        return

    def read_output(self):
        output = ''
        with open(self.file_dir / self.output_file, 'r') as fil:
            for line in fil:
                output += line
        return output

    def write_input(self):
        str_input = ''.join([str(i) for i in self.input])
        self.input_length = len(str_input)
        with open(self.file_dir / self.input_file, 'w+') as fil:
            fil.write(str(str_input))

    def write_seed(self):
        str_seed = ''.join([str(i) for i in self.seed])
        with open(self.file_dir / self.seed_file, 'w+') as fil:
            fil.write(str(str_seed))

    def execute_extractor(self):
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
