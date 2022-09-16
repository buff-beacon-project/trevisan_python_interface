'''
#!/usr/bin/env python
# coding: utf-8

# # Randomness Beacon Software Guide: An Overview
  ### Aliza Urooj Siddiqui
# 
# The Randomness Public Beacon project is an idea created by the National Institute for Standards and Technology (NIST) to help generate truly random binary bits with a security certificate. This beacon will be for public use and is applicable to any scenario which needs unbiased randomness. 
# Some potential applications include (but not limited to):
#    * Lottery games
#    * Prevention of Gerrymandering
#    * Casino games
#    * Seeding of private randomness sources for communication key generation
# 
# The beacon or pipeline has three main components:  
# **Experimental**   
# **Analysis**  
# **Extractor**
# 
# Each phase serves a special purpose. Let's briefly go through each one
# 
# ## Experimental Phase
# *NOTE: This explanation is very to-the-point and by no means explains all the tiny, intricate details of the experiment. Purpose of this guide is to focus on the analysis component of the pipeline
#  More information can be found here: https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.115.250402
# 
# To create this beacon, NIST utilizes the fact that true randomness can be found within nature itself that cannot be explained by classical or intuitive reasons--this is known as the quantum (subatomic) realm. Nobody could replicate the outcomes in nature by simply altering one or two variables, like temperature.
# Long ago, an experiment was formulized to "prove" this fact known as the Bell Experiment. Some of Bell Experiment's results go against any prediction or local theory that individuals can come up with to potentially explain those results. 
# So, in the proper environment (loophole free), NIST conducts this experiment and utilizes these results for the beacon. 
# The Bell Experiment is as follows:
# 1. A source generates an entangled pair of particles which are sent in opposite directions, one to Alice and the other to Bob
# 2. Each party picks between two possible measurement settings (independent from one another and determined by a Random Number Generator) 
# 3. Once the measurement setting is picked, each party measures the particle and the detector for each determines if the particle arrives (was detected) or not
# 4. After the experiment is done, both parties compare their outcomes with each other
# 
# The data results for each trial conducted for this experiment follows the following format: 
# 
# $S_{A} S_{B} O_{A} O_{B}$
# 
# Once the experiment is ran and the experimental data is converted to the aforementioned format, the latter is handed down to the analysis component
# 
# ## Analysis Phase
# 
# The Analysis phase focuses on analyzing the Bell Experiment data and quantifying how much entropy (randomness) is in the results. The entropy amount must reach a certain threshold (determined before analysis) in order to generate k random bits. The threshold value depends on how many random bits you desire. 
# In our case, for 512 random bits, our entropy threshold must reach $\sigma$ = 1089
  
# To analyze the data, NIST utilizes the probability estimation framework
   *You must place different constraints on data such as no signaling constraints (Alice and Bob could not influence each other's settings choices)
    positive constraints, etc.
   *You have to calculate Probability Estimation Factors which are basically ratios between the probability of getting an experimental result in a world dominated by Local Realism
    vs in a world dominated by Quantum Mechanics (unpredictability)
   *Every experimental outcome is assigned a specific PEF (16 total) and for every outcome we observe in our experiment, we chain the PEFs (multiply them) together
   * The final value from our chain is our p-value whose value determines the strength of our null hypothesis being true
   NOTE: The null hypothesis states that local realism indeed is the explanation behind our natural world. There are hidden variables that determine every outcome
   The smaller our p-value is, the less likely our null hypothesis is actually true
   
   So, we try to prove the hypothesis false by trying to get a small p-value
  
   * After the protocol is done, we determine if we reached our entropy threshold or not by quantifying how much entropy we accumulated from the data
   
# 
# Let's get into the software and how to run it
# 
# In general, the modules will be ran in the following order:  
# * **PEF_Analysis**: This module will set the basic parameters before the experiment is began (e.g Number of settings/outcomes, entropy threshold, etc.  
# * **Data_Loading_Mod**: Data loading modified will be in charge of taking in binary data files and extracting the data inside  
# * **PEF_Calculator**: This module will calculate the probabiliy estimation factors and the gain we expect out of the data files.
# * **PEF_Accumulator**: This module will aggegrate entropy from randomness analysis binary data files after calibration is done and PEFs have been calculated  
# 
# 

#First let us import all the software modules needed:
'''

# In[5]:


from data_loading_mod import Data_Loading_Mod
from PEF_Accumulator import PEF_Accumulator
from PEF_Calculator import PEF_Calculator
from PEF_Analysis import PEF_Analysis
import numpy as np
import os as os


'''
FIRST STEP: Initialize Variables
            * These include the entropy thresholds, biases, file path, etc.
____________________________________________________________________________
'''

path = r'E:\2018_04_13_all_files_processed\processed\Experimenting'
beta = 0.01
epsilon_bias = 0.001
delta = 4e-8
entropy_threshold = 1089
current_pefs = np.zeros((16,))
current_exp_gain = 0
numFiles = 10



'''
SECOND STEP: Create Entropy Accumulator Object
            * This object will be in charge of accumulating randomness (entropy)
              for one set of 512 bits
____________________________________________________________________________
'''
# Now that the modules are imported, path to the data files is declared, and variables are initialized, we will create an entropy accumulator object

pef_accumulator = PEF_Accumulator(path, beta, epsilon_bias, delta) 


'''
THIRD STEP: Calibration - Creating PEFs 
            * You must initialize the starting index where the calibration will
              began
            * You must also create a data loading object which will be used 
              specifically for calibration (another one will be created for randomness
              generation later)
____________________________________________________________________________
'''
#Also, the starting index for randomness accumulation is the file right after the last file used for calibration (calculation of PEFs)
#numFiles were the number of files used for calibration

starting_index = numFiles - 1 #start after the index of the last calibration file
dataLoading_calib = Data_Loading_Mod(path, numFiles) #numFiles is how many files you want to read in when it comes time to recompute PEFs/calibration

'''
(3.1) Data Processing Function: 
    The data_processing(...) function puts it all together and starting from a 
    starting_index, aggregates frequencies from a certain number of binary data files, 
    given by the parameter numFiles. This method is mainly used for the calibration step of the experiment. 
    In other words, if your experiment is running for a long time or starts to give bad data, 
    you can readjust your experiment to fix any issues.
    
    INPUT: The starting index
    OUTPUT: The 4x4 frequencies matrix with all the S + O from all the calib binary files
'''
print("Update PEFs/ Creating PEFs")
freq = dataLoading_calib.data_processing(starting_index) #starting is the starting index for calibration: data files will be read backwards. 
print("Freq being given to calc pef and gain: ", freq)


'''
(3.2) Calc PEF and Gain Function: 
     The calc_PEF_and_gain(...) function calculates our probability estimation factors 
     that will use to calculate our entropy in our data
     It also calculates the gain we expect to recieve from our data
     
    INPUT: Our beta power parameter and our epsilon bias in our settings distributions 
    OUTPUT: PEFs 16x1 and our expected gain value
'''

'''
First create a PEF Calculator Object. A new object will be created for every set of PEFs you wish to generate
Meaning, if you wish to recompute the PEFs at any point while your experiment is ongoing, a new PEF_Calculator object is created
This object is NOT specific to the set of 512 bits you wish to create but how many times the PEFs are computed 
'''
pef_calculator = PEF_Calculator(freq.T, path, beta, epsilon_bias, delta)
pefs, exp_gain = pef_calculator._calc_PEF_and_gain(beta, epsilon_bias) 



'''
(3.3) Compute LR Stat Strength Function: 
    The compute_LR_stat_strength(...) function computes the statistical strength 
    of given frequencies with respect to the KL divergence and the LR polytope.
    Basically determines if the PEFs we generate are "good enough"
    
    INPUT: The frequencies aggregated from the calib files
    OUTPUT: The statistical strength
'''
pef_calculator.compute_LR_stat_strength(freq.T)

'''Here we are storing the PEFs and Gain in global variables
# NOTE: I am aware that global variables could cause issues however these are not 
necessary in your final code
'''
print("pefs calculated: " , pefs)
current_pefs = pefs
print("current_pefs: " , current_pefs)
current_exp_gain = exp_gain
print("current_exp_gain: " , current_exp_gain)
        
'''
FOURTH STEP: Randomness Generation- Accumulating Randomness 
            * Now we are getting into the actual randomness accumulation
            * Through this last step, we will decide if the protocol succeeded to
              meet our entropy threshold or failed
            * NOTE: The current code has a dummy variable as the minimum number of trials
              that the protocol is allowed to use to try to reach \sigma
              ~~~ Be careful! We call this the "minimum number of trials" we expect the protocol
              to need in order to reach our threshold but it actually acts as a maximum
              number of trials the protocol can use.
              If the protocol reaches this number of trials (binary files) before reaching
              the entropy threshold, we know the protocol instance failed~~~
              So...in other words...this parameter is also pretty important!
____________________________________________________________________________
'''
#First create a second data loading object which is specific to randomness generation
trial_interval = 5
min_trials = 4
current_freq = np.zeros((4,4)) 
data_loading_rand = Data_Loading_Mod(path, min_trials)
trials = 0 #trial counter - will keep track of how many trials (data files) have been used
files = os.listdir(path) 

#This is the loop that will keep track of the number of trials (files) used by the protocol and make sure
# that the protocol does not go over the min_trials it is allowed
while trials < min_trials:
                '''
                (4.1) read_data_file Function:
                      This function reads a binary data file at a specific index in a certain format (SA,SB,OA,OB)
                      INPUT: the binary file
                      OUTPUT: the data in the aforementioned format
                '''
                data = data_loading_rand.read_data_file(files[starting_index])
                
                '''
                (4.2) get_freqs Function:
                      Currently each trial in the data file is stored as an element in the data list in the format 
                      (SA, SB, OA, OB). The goal of this function is to create a 4x4 frequencies matrix where each element 
                      is a combination of the settings + outcomes for Alice and Bob. 
                      For example, row 1 column 1 of the matrix would be ab00, row 1 column 2 would be ab01, etc.
                      
                      INPUT: the binary data in SA, SB, OA, OB
                      OUTPUT: 4x4 matrix of S + O for that specific binary file's results
                '''
                freq = data_loading_rand.get_freqs(data) #get frequencies matrix of the data file
                print("freq from data file: ", freq)
                
                current_freq = np.add(current_freq, freq) #add those frequencies to overall frequencies matrix (from all processed data files so far)
                print("current_freq: ", current_freq)
                
                '''
                (4.3) accumulate_entropy Function:
                      Utilizes the following formula to calculate the entropy accumulated 
                      given a matrix of frequencies and current pefs:
                      
                      sum(sum(freq*log2(pefs/(1+ delta))))/beta 
                      
                      This formula is also in the MATLAB Analysis Code line 29
                      of prep_randomness_analysis.m
                      INPUT: the binary data in SA, SB, OA, OB
                      OUTPUT: 4x4 matrix of S + O for that specific binary file's results
                '''
                entropy = pef_accumulator.accumulate_entropy(current_freq, current_pefs) #use overall freqs so far and associated pefs to accumulate entropy
                print("Entropy accumulated so far: ", entropy)
                
                
                trials = trials + 1  #Increments the number of trials and keeps track
                starting_index = starting_index + 1 #Increments the starting index to read in the next binary file in the path
                
                '''
                Here we see if our trials have reached a certain interval.
                If our experiment is running for a long time, there are often technology drifts and
                we need to recalibrate to get better results.
                So, after a certain amount of time (after a certain amount of binary files (trial interval)), 
                we update our PEFs to new values (recompute them)
                NOTE: The experimentalists may determine what trial interval is best and how often the PEFs need to be
                      updated
                '''
                if trials % trial_interval == 0: #time to update pefs
                    print("In Update PEFs")
                    freq = dataLoading_calib.data_processing(starting_index - 1) #starting is the starting index for calibration: data files will be read backwards. 
                    print("Freq being given to calc pef and gain: ", freq)
                    pef_calculator = PEF_Calculator(freq.T, path, beta, epsilon_bias, delta)
                    pef_calculator.compute_LR_stat_strength(freq.T)
                    print("self.beta: " , beta)
                    print("self.epsilon_bias: ", epsilon_bias)
                    pefs, exp_gain = pef_calculator._calc_PEF_and_gain(beta, epsilon_bias) 
                    print("pefs calculated: " , pefs)
                    current_pefs = pefs
                    current_exp_gain = exp_gain
                    print("current_exp_gain: " , current_exp_gain)
                
                '''
                  Here we check whether our current accumulated entropy has reached our set entropy threshold
                  or not.
                  If so, we can stop early! No need to go all the way to min_trials
                  
                '''
                    
                if(entropy >= entropy_threshold): #if accumulated entropy is above the threshold required to generate k bits, stop data aggregation
                    print("Protocol Successful!\n")
                    print("Reached after ", trials, " trials")
                    print("Entropy accumulated: ", entropy)
                    print(entropy)
                    break

'''
We are now outside the while loop....
Meaning, unfortunately we were unable to reach our entropy threshold
before reaching the minimum number of trials our protocol is allowed

So...our protocol failed and we return how much entropy we *were* able to
accumulate so we can see how far we were
                  
'''                   
if(entropy < entropy_threshold):
    print("Protocol Failed. Maximum trials reached\n")
    print("Only able to accumulate: ", entropy)
    print(entropy)
      
'''
So, that is it, folks!
This is the order in which you must run the modules
NOTE: we did not explicitly create an object of PEF Analysis and run that module due to class inheritance
The modules, PEF Calculator, Data Loading, and PEF Accumulator all extend the module and utilize its instance variables so 
it is automatically initialized.

So, all this might look very complicated at first glance. 
Good news is that I have created a Manager class as a template for you all. Inside, the process of updating the PEFs is 
packaged inside a function, update_PEFs and the randomness generation is packaged inside the function, file_processing()

THIS MODULE WILL BE YOUR POINT OF CONTACT BETWEEN YOURSELF AND THE SERVICE MODULES: PEF Accumulator, PEF Calculator, Data Loading
Feel free to edit this module as you wish.
You should not have to alter anything inside the service modules
   

Best of Luck!
'''

# In[ ]:




