import functools
from datetime import datetime
from statistics import mean, stdev
#from numba import jit
import numpy as np
import scipy
import time
from PEF_Calculator import PEF_Calculator
from PEF_Analysis import PEF_Analysis
import scipy.io
import os as os

#path = r'E:\2018_04_13_all_files_processed\processed\Experimenting'


class PEF_Accumulator(PEF_Analysis):
    """class for accumulating actual gain(entropy)
    """

    def __init__(self, beta, epsilon_bias, delta, **kwargs):
          super().__init__(beta, epsilon_bias, delta,  **kwargs)
          self.entropy = 0


    def get_current_entropy(self):
        """ @author: Aliza Siddiqui
        Returns the current entropy that has been accumulated so far
            :return - the entropy
        """
        return self.entropy


    #Calculates the entropy accumulated
    #Note I have split up the terms into parts just to make sure the results are accurate
    def accumulate_entropy(self, freq, pefs):
        """ @author: Aliza Siddiqui
        Utilizes the following formula to calculate the entropy accumulated given a matrix of frequencies and current pefs
        sum(sum(freq*log2(pefs/(1+ delta))))/beta
        This formula is also in the MATLAB Analysis Code line 29
        of prep_randomness_analysis.m
            :param freq: current frequencies to use
            :param pefs: current pefs to use
            :return - the entropy
        """
        pefs = self.convert_pef_matrix(pefs)
        log_pefs = np.log2(pefs/(1+(self.delta)))
        first_term = np.multiply(freq, log_pefs)
        first_term = first_term.sum(axis=0)
        first_term_final = first_term.sum()
        #second_term = np.log2(((4.34e-20)**2)/2.0) #Rescale for QEF
        entropy = (first_term_final)/self.beta
        self.entropy = entropy
        return entropy

    def convert_pef_matrix(self, pefs):
        """ @author: Aliza Siddiqui
        Changes the shape of the pef matrix from (16,1) to (4,4) matrix for easy matrix calculations
            :param pefs: current pefs to use
            :return - the modified shape pefs matrix
        """
        pefs = pefs.reshape((4,4))
        pefs = pefs.T
        return pefs
