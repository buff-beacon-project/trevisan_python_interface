import numpy as np
import itertools


class PEF_Analysis(object):
    """This is the base class for quantum randomness generation via probability estimation. It was specifically written for analysis of data collected at NIST in Aug, 2018.
    """
    #creating different models for probability distributions that could arise
    def __init__(self, beta, epsilon_bias, delta , stat_strength_threshold = 10000, entropy_threshold = 1089, n_pulses=1, n_outcomes_a = 2, n_outcomes_b = 2, n_settings_a = 2, n_settings_b = 2):
        self.n_pulses = n_pulses
        print("Number of pulses", n_pulses) # number of pulses is a currently necessary parameter to physically enforce space-like separation of the two labs.
        self.n_settings_a = n_settings_a
        print("Number of Settings (Alice)", n_settings_a)
        self.n_settings_b = n_settings_b
        print("Number of Settings (Bob)", n_settings_b)
        self.n_outcomes_a = n_outcomes_a
        print("Number of Outcomes (Alice)", n_outcomes_a)
        self.n_outcomes_b = n_outcomes_b
        print("Number of Outcomes (Bob)", n_outcomes_b)
        self.n_outcomes = self.n_outcomes_a * self.n_outcomes_b
        print("Number of Total Outcomes", self.n_outcomes_a * self.n_outcomes_b)
        self.n_settings_choice = self.n_settings_a * self.n_settings_b
        print("Number of Total Settings Choices",  self.n_settings_a * self.n_settings_b)
        self.beta = beta
        print("Beta Parameter", beta)
        self.epsilon_bias = epsilon_bias
        print("Epsilon Bias Parameter in Settings: ", epsilon_bias)
        self.delta = delta
        print("Delta Parameter: ", delta)
        self.stat_strength_threshold = stat_strength_threshold
        print("Statistical Strength Threshold", stat_strength_threshold)
        self.entropy_threshold = entropy_threshold
        print("Entropy Threshold", entropy_threshold)


        # Construct the model of probability distributions for the implementation. Nominally, This consists of non-signaling behaviors subject to quantum constraints such as the Tsirelson bound.
        # For the (2,2,2) case, the non-signaling model consists of local realistic models (LR) extrema and Popescu-Rohrlich (PR) box extrema.
        self._create_LR_Matrix()
        self._create_PR_Matrix()  #PR Boxes: stronger-than-quantum bipartite correlations, violation of relativistic causality
        self._create_I_CHSH()




    def _create_LR_Matrix(self):
        """ @author: Mohammad Alhejji
        Constructs the matrix of (LR) extrema for the general (2, n_outcomes_a, n_settings_choice_a). There are 16 possible
        local realistic outcomes for the (2,2,2) case.
        """
        n = self.n_outcomes * self.n_settings_choice
        self.mLR = np.zeros((n, n))
        local_strat_num = 0
        for p in range(self.n_pulses):
            # Each LR vector is determined by two independent functions f_A and f_B
            # mapping settings x,y to outcomes f_A(x) and f_B(y) respectively.
            for f_A in [{setting: outcome for (setting, outcome) in zip(range(self.n_settings_a), outcomes)}
                        for outcomes in itertools.product(range(self.n_outcomes_a), repeat=self.n_settings_a)]:
                for f_B in [{setting: outcome for (setting, outcome) in zip(range(self.n_settings_b), outcomes)}
                            for outcomes in itertools.product(range(self.n_outcomes_b), repeat=self.n_settings_b)]:
                    # Col = 'f_A(x)f_B(y)xy' is nonzero for settings 'xy' and local strategies f_A, f_B
                    for x in range(self.n_settings_a):
                        for y in range(self.n_settings_b):
                            self.mLR[local_strat_num][
                                (((f_A[x]*self.n_outcomes_b + f_B[y])*self.n_settings_a) + x)*self.n_settings_b + y] = 1
                    local_strat_num += 1
        self.mLR = np.mat(self.mLR, dtype = np.longdouble)


    #creates PR Boxes (2, 2, 2) 2 stations, 2 outcomes for Alice
    def _create_PR_Matrix(self):
        """ @author: Mohammad Alhejji
            @Last modified: 07/16/2020 by Aliza Siddiqui
        Constructs the matrix of PR boxes for the specific case (2,2,2). There are 8 of these.
        8 possible outcomes for PR Boxes (very nonlocal)
        """
        n = self.n_pulses * self.n_outcomes * self.n_settings_choice
        m = self.n_pulses * (2 ** (self.n_settings_choice - 1))

        mPR = np.zeros((m, n))
        pr_extrema = 0
        for p in range(self.n_pulses):
            # Each row is an extrema of the pr_box given by a binary function g with
            # an odd number of outputs g(xy) being 1
            # Here g is a string with character at index xy  is g(xy)
            #     I don't believe this generalizes to more than 2 possible outcomes
            for g in range(2 ** (self.n_settings_choice - 1)):
                g = bin(g)[2:]
                if g.count('1') % 2 == 1:
                    g += '0'
                else:
                    g += '1'
                g = '0' * (self.n_settings_choice - len(g)) + g

                for sA in range(self.n_settings_a):
                    for sB in range(self.n_settings_b):
                        settings_num = (sA * self.n_settings_b) + sB
                        g_val = int(g[settings_num])
                        for oA in range(self.n_outcomes_a):
                            for oB in range(self.n_outcomes_b):
                                if oA == (oB + g_val) % 2:
                                    ind_AB = ((oA * self.n_outcomes_b) + oB)
                                    mPR[pr_extrema][(ind_AB * self.n_settings_choice) + settings_num] = 1 / 2.



                '''for sA, sB in product(range(self.n_settings_a),range(self.n_settings_b)):
                        settings_num = (sA * self.n_settings_b) + sB
                        g_val = int(g[settings_num])
                        for oA, oB in product(range(self.n_outcomes_a),range(self.n_outcomes_b)):
                                if oA == (oB + g_val) % 2:
                                    ind_AB = ((oA * self.n_outcomes_b) + oB)
                                    mPR[pr_extrema][(ind_AB * self.n_settings_choice) + settings_num] = 1 / 2.'''
                pr_extrema += 1
        self.mPR = np.mat(mPR, dtype  = np.longdouble)

    def _create_I_CHSH(self):
        """ @author: Mohammad Alhejji
        Create the CHSH inequalities.CHSH inequalitites are only used for (2,2,2) scenario bell tests. These are also used to impose Tsirelson's bound.
        """
        #creates the CHSH inequalities; this is the model
        mCHSH = np.zeros((4, 16))

        for anti_correlation_idx in range(4):
            for oA in [0, 1]:
                for oB in [0, 1]:
                    for sA in [0, 1]:
                        for sB in [0, 1]:
                            settings_num = (sA << 1) + sB
                            correlate = 2 * (oB == oA) - 1
                            should_correlate = 2 * (settings_num != anti_correlation_idx) - 1
                            sign = correlate * should_correlate
                            index = (oA << 3) + (oB << 2) + (sA << 1) + sB
                            mCHSH[anti_correlation_idx][index] = sign
        self.mCHSH = np.array(np.vstack((mCHSH, -mCHSH)), dtype = np.longdouble)
        #be careful with the precision of the numbers you enter in python; you may want to use long double
