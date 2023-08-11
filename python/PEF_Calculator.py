import os
import time
import itertools
import functools
import math
import numpy as np
import scipy.optimize
import scipy.interpolate
from scipy.optimize import fsolve
# import matplotlib
# import matplotlib.pyplot as plt
# from matplotlib import colors
from PEF_Analysis import PEF_Analysis
import cvxpy as cvx
import primefinding as prime

np.set_printoptions(precision=128, threshold=10**15)


class PEF_Calculator(PEF_Analysis):
    """At the beginning, this class is to be instantiated to calculate the probability estimation factors.
    """
    def __init__(self, frequencies, beta, 
                    epsilon_bias=0, delta=0, 
                    default_setting=0,
                    **kwargs):
        super().__init__(beta,  epsilon_bias, delta, **kwargs)
        self.beta = beta
        self.epsilon_bias = epsilon_bias
        self.delta = delta
        self.default_setting = default_setting

        self.N = 16
        self.shape = (4,4)

        self.dtype = np.float128
        self.eps = np.finfo(np.float64).eps

        
        self.set_frequencies(frequencies)
        self.update_pefs()

    def set_frequencies(self, frequencies):
        freqNorm, conditional_freqs = self._process_frequencies(frequencies)
        self.frequencies = freqNorm
        self.conditional_freqs = conditional_freqs
        self.setup()

    def update_pefs(self):
        pefs, gain = self._calc_PEF_and_gain()
        self.pefs = pefs
        self.gain = gain

    def _process_frequencies(self, frequencies):
        # Find conditional frequencies consistent with non-signaling.
        freq = frequencies.reshape(self.shape[0],self.shape[1])
        freqNorm = self.normalize_freq(freq)
        conditional_freqs = self.compute_ns_conditional_frequencies(freqNorm)
        return freqNorm, conditional_freqs

    def normalize_freq(self, freq):
        # freqNorm = freq/np.sum(freq, axis=0, keepdims=True)
        freqNorm = freq*1./np.sum(freq)
        return freqNorm

    def setup(self):
        self.pefs = np.zeros((16,))
        # Account for violated CHSH inequality (if one is violated)
        I_CHSH = self._find_opt_I_CHSH(self.conditional_freqs)
        V=self.dtype(np.sqrt(2) - 1) #why is the violation sqrt(2) -1 and not sqrt(2) -2?
        A = self.mLR[np.array(self.mLR * I_CHSH.transpose(), dtype=self.dtype).flatten() == 2, :]
        B = self.mPR[np.array(self.mPR * I_CHSH.transpose(), dtype=self.dtype).flatten() == 4, :]
        self.C = (1 - V) * A + V * np.tile(B, (len(A), 1))

        # This is set of constraints in addition to non-negativity is the model assumed. Note that it is strictly bigger than Q and strictly smaller than NS.
        self.ns_vectors = np.concatenate((
            self.mLR,
            self.C,
            self.mPR[np.array(self.mPR * I_CHSH.transpose(), dtype=np.longdouble).flatten()!= 4, :]))


    def compute_ns_conditional_frequencies(self, frequencies):
        """ @author: Mohammad Alhejji
        Computes the best conditional frequencies consistent with non-signaling constraints
            :param frequencies - The raw frequencies
            :return - The conditional frequencies
        """
        freq = frequencies.reshape(self.N) #freq is now a 1x16
        # freq = freq/np.sum(freq)  #Normalize

        cond_freq_mat = cvx.Variable(self.shape)
        cond_freq_var = cvx.reshape(cond_freq_mat.T, (self.N, 1)) #16x1

        obj = cvx.Minimize(-freq @ cvx.log(cond_freq_var))

        alice_settings = [
            np.array([setting // self.n_settings_b == setting_a for setting in range(self.n_settings_choice)])
            for setting_a in range(self.n_settings_a)]
        bob_settings = [
            np.array([setting % self.n_settings_a == setting_b for setting in range(self.n_settings_choice)])
            for setting_b in range(self.n_settings_b)]
        alice_outcomes = [
            np.array([outcome // self.n_outcomes_b == outcome_a for outcome in range(self.n_outcomes)]).reshape(
                (1, self.n_outcomes)
            )
            for outcome_a in range(self.n_outcomes_a)]
        bob_outcomes = [
            np.array([outcome % self.n_outcomes_a == outcome_b for outcome in range(self.n_outcomes)]).reshape(
                (1, self.n_outcomes)
            )
            for outcome_b in range(self.n_outcomes_b)]

        # Alice's outcome should be independent of Bob's setting
        a_ns_constraints = [outcome_a@cond_freq_mat[:, bob_settings[0]] == outcome_a@cond_freq_mat[:, setting_b]
                            for setting_b in bob_settings[1:] for outcome_a in alice_outcomes]
        # Bob's outcome should be independent of Alice's setting
        b_ns_constraints = [outcome_b@cond_freq_mat[:, alice_settings[0]] == outcome_b@cond_freq_mat[:, setting_a]
                            for setting_a in alice_settings[1:] for outcome_b in bob_outcomes]

        # Tsirelson bound + NS conditions
        I_CHSH_all = self.mCHSH
        tsirelson_bound_constraints = [
            I_CHSH_all@cond_freq_var <= 2*np.sqrt(2) # There are 8 variants because CHSH is between -2 and 2.
        ]
        constraints = [cond_freq_mat >= 0,
                      cvx.sum(cond_freq_mat, axis=0) == 1,
        ] + a_ns_constraints + b_ns_constraints + tsirelson_bound_constraints

        prob = cvx.Problem(obj, constraints)
        prob.solve('ECOS', abstol=1e-15)
        # prob.solve(solver=solver)#, abstol=1e-15)
        cond_freq = np.array(cond_freq_mat.value).reshape(self.N)
        return cond_freq



    def _find_opt_I_CHSH(self, cond_freqs):
        """ @author: Mohammad Alhejji
        Computes the CHSH conditions that the conditional frequencies violate
            :param cond_freq - The conditional frequencies
            :return - The CHSH models that need to be considered
        """
        I_CHSH_models = np.mat(self.mCHSH)
        #  Which CHSH inequalities are violated
        opt_CHSH_ind = np.argwhere((I_CHSH_models * np.mat(cond_freqs).transpose()) > 2)[:, 0]
        I_CHSH = I_CHSH_models[opt_CHSH_ind, :]
        #  There should be at most one violated inequality which we need to account for in the contstraints
        assert len(I_CHSH) <= 1
        # print("CHSH Models to be Considered", np.mat(I_CHSH))
        return np.mat(I_CHSH)



    def calculate_convex_constraints(self, settings_bias, beta, num_vars):
        """ HELPER FUNCTION TO _calc_PEF_and_gain
            @author: Aliza Siddiqui
        Computes the convex constraints for a set of settings_bias in: [p**2, p*q, p*q, q**2], [p*q, q**2, p**2, p*q],
        [p*q, p**2, q**2, p*q], and [q**2, p*q, p*q, p**2]
            :param settings_bias - epsilon_b parameter
            :param beta - power for the experiment
            :num_vars - number of variables
            :return - The convex constraints
        """
        # settings_weight should be a probability distribution
        # settings_bias /= sum(settings_bias)

        #Constraints matrix given the specific probability distribution on the convex polytope
        constraints_mat = (1)*np.multiply(
            np.power(self.ns_vectors, 1 + beta),
            np.tile(settings_bias, (len(self.ns_vectors), self.n_outcomes)))

        #Convex_constraints given the specific constraints_mat
        convex_constraints = {
            'type': 'ineq',
            'fun': lambda x: 1 - np.array(constraints_mat * np.mat(x).reshape((num_vars, 1))).flatten(),
            'jac': lambda x: - constraints_mat
            }
        return convex_constraints


    def place_convex_constraints(self, epsilon_bias, beta, num_vars):
        """ HELPER FUNCTION TO _calc_PEF_and_gain
            @author: Aliza Siddiqui
        Calculates all four probability distributions on the convex polytope given a certain epsilon and
        computes the convex constraints for all 4: [p**2, p*q, p*q, q**2], [p*q, q**2, p**2, p*q],
        [p*q, p**2, q**2, p*q], and [q**2, p*q, p*q, p**2]
            :param epsilon_bias - How much bias there is in the settings (RNGs)
            :param beta - power for the experiment
            :num_vars - number of variables
            :return - The convex constraints for each probability distribution
        """
        #D(Z) of distributions of Z(XY) is a convex polytope with 4 extreme points.
        # At the extreme points, the probability distributions are given by:
        p = 0.5 + epsilon_bias
        q = 1-p

        settings_bias1 = np.array([p**2, p*q, p*q, q**2], dtype=self.dtype)
        settings_bias2 = np.array([p*q, q**2, p**2, p*q], dtype=self.dtype)
        settings_bias3 = np.array([p*q, p**2, q**2, p*q], dtype=self.dtype)
        settings_bias4 = np.array([q**2, p*q, p*q, p**2], dtype=self.dtype)

        #Calculating the convex constraints for each probability distribution
        convex_constraints1 = self.calculate_convex_constraints(settings_bias1, beta, num_vars)
        convex_constraints2 = self.calculate_convex_constraints(settings_bias2, beta, num_vars)
        convex_constraints3 = self.calculate_convex_constraints(settings_bias3, beta, num_vars)
        convex_constraints4 = self.calculate_convex_constraints(settings_bias4, beta, num_vars)

        convex_constraints = [convex_constraints1, convex_constraints2, 
                                convex_constraints3, convex_constraints4]

        return convex_constraints

    def optimizing_PEFS(self, objf, x, jac, hess, constraints, tol, iter_limit):
        """ HELPER FUNCTION TO _calc_PEF_and_gain
            @author: Aliza Siddiqui
        Optimizes PEFS using the convex and positive constraints for a certain guess x
            :param objf - The objective function
            :param x - optional starting point for optimization
            :param constraints - convex constraints, positive constraints, and the trivial constraints
            :param tol - tolerance for objective function
            :iter_limit - maximum number of iterations
            :return - The convex constraints
        """
        res = scipy.optimize.minimize(
            objf,
            x,
            method='SLSQP',
            jac=jac,
            constraints=constraints,
            tol = 1E-12,
            options={
                'disp': False,  # disabled, here for debugging purposes
                'ftol': tol,
                'maxiter': iter_limit
            })

        return res


    def norm_pefs(self,pos_result, constraints):
        """ HELPER FUNCTION TO _calc_PEF_and_gain
            @author: Aliza Siddiqui
        Makes extra sure that the constraints are satisfied and gives the PEFS
            :param pos_results - The objective function
            :param convex constraints - the convex constraints for a certain probability distribution
            :param epsilon -
            :return - The PEFs
        """
        correction_vals = []
        for constraint in constraints:
            # print(constraint)
            cv = (1-min(constraint['fun'](pos_result)) + self.eps)
            correction_vals.append(cv)
        correction_val = max(correction_vals)
        
        pef= pos_result/correction_val
        return pef

    def _calc_PEF_and_gain(self):
            """ @author: Mohammad Alhejji
                @Last Modified: 07/22/2020 by Aliza Siddiqui

            Calculates the pef and gain for a given setting_weight

              :param beta: beta parameter (>0)
              :param epsilon_bias: bias in the settings choices
              :param guess: optional starting point for optimization.
              :param iter_limit: maximum number of iterations
              :param default_setting: the default setting for the experiment
              :param tol: tolerance for objective function

              :return: (PEF, gain)
            """
            beta = self.beta
            epsilon_bias = self.epsilon_bias 
            guess = None 
            iter_limit = 10000
            default_setting = self.default_setting
            tol = 1E-12
            num_vars = self.n_outcomes * self.n_settings_choice
            freq = self.conditional_freqs.reshape(num_vars)

            scale = 1
            freq *= scale

            convex_constraints =  self.place_convex_constraints(epsilon_bias, beta, num_vars)
        # Objective function and gradient
            objf = lambda x: self._log_likelihood(x, freq)
            jac = lambda x: self._log_likelihood_grd(x, freq)
            hess = lambda x: self._log_likelihood_hess(x, freq)

        # PEFs are positive
            positive_constraints = [{
            'type': 'ineq',
            'fun': lambda x: x,
            'jac': lambda x: np.mat(np.eye(num_vars))
            }]

        # PEFs for no-click no-click are set to 1 for the default setting
            ind00 = [default_setting]
            trivial_pef_constraints = [{
                  'type': 'eq',
                  'fun': functools.partial(lambda idx, x: x[idx] - 1, ind),
                  'jac': functools.partial(
                   lambda idx, x: np.hstack((np.zeros((1, idx)), np.eye(1), np.zeros((1, num_vars-1-idx)))),
                   ind)
                   } for ind in ind00]


            x0 = guess if guess is not None else np.array([1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1], dtype=self.dtype)

            # x1 = np.array([0.999985100015945,1.000014959703430,1.000014959703430,0.999984980337838, 
            #          0.960053330288753,0.996179015633874,0.929773555664518,1.003805611257416, 
            #          0.961278860973820,0.928539989152853,0.996567940251360,1.003418239233214, 
            #          1.031270546920231,1.034730739709108,1.036340302673597,0.897122388776918])
            
            pos_results = []
            constraints = convex_constraints + positive_constraints + trivial_pef_constraints
            res = self.optimizing_PEFS(objf,  x0, jac, hess, constraints, tol, iter_limit)
            pefTmp = res.x
            pefTmp[pefTmp<0] = 0
            pos_results = [pefTmp]

            pefs = self.norm_pefs(pos_results[0], convex_constraints)
            gain = self.calc_gain(pefs, self.frequencies, beta)

            # print("pefs: ", pefs)
            # print("Expected Gain: ", gain)
            return pefs, gain #returns 16x1 array of the optimal PEFs as well as a decimal value of the gain

    def calc_gain(self, PEF, freq, beta):
        # sh = np.shape(freq)
        # N = sh[0]*sh[1]
        freq = freq.reshape(self.N)
        logPEFs = np.log2(PEF).reshape(self.N)
        expectation = np.dot(freq,logPEFs)
        gain = expectation/beta
        return gain

    def _log_likelihood(self, x, freq):
        """
          @author: Mohammad Alhejji
        Computes the log likelihood of x
        Solves the following: Max{ E_v(log_2(PEF)) }
          :param x:
          :param freq:
          :return: log likelihood
        """
        x = x[freq > 0]
        if min(x) < 1e-20:
            return np.inf
        freq = freq[freq > 0]
        lg = np.log2(x, dtype=self.dtype)
        res = -freq.dot(lg)
        return res

    def _log_likelihood_grd(self, x, freq):
        """
          @author: Mohammad Alhejji
        Computes the gradient of the log likelihood (the fastest rate at which log likelihood is increasing)
          :param x:
          :param freq:
          :return:  gradient
        """
        gradient = np.zeros(len(freq))
        gradient[freq > 0] = - np.array(freq[freq > 0])*1./(np.array(x[freq > 0]))
        gradient /= np.log(2)
        return gradient

    def _log_likelihood_hess(self, x, freq):
        """
          @author:Krister Shalm
        Computes the Hessian of the log likelihood 
          :param x:
          :param freq:
          :return:  gradient
        """
        N = len(freq)
        hess = np.zeros(N)
        hess[freq>0] = np.array(freq[freq>0])/np.array(x[freq>0]**2)
        hess /= np.log(2)
        maxHess = np.max(hess)
        hess[hess==0] = np.abs(maxHess)+1000.
        hessMatrix = np.diag(hess)

        return hessMatrix


    def PEF_vio_check(self, settings_weight, beta, PEFs, model = None ):
        """
          @author: Mohammad Alhejji
        Checks the PEF violation of some given model at a given beta at a given input choices. The default model
        is no-signaling with Tsirelson's bound.
          :param model: set of extrema of model of behaviors
          :param settings_weight: input distribution to the box (e.g. measurement choices)
          :param beta: exponent for the pefs
          :param PEFs: estimation factors
          :return: f_max, the biggest RHS value
        """
        if model is None:
            model = self.ns_vectors

        if sum(settings_weight) > 1 + self.eps:
            return('settings_weight must be normalized')
        # list of constraints
        constraints_mat = (1)*np.multiply(
            np.power(model, 1 + beta),
            np.tile(settings_weight, (len(model), self.n_outcomes)))

        convex_constraints = {
            'type': 'ineq',
            'fun': lambda x: np.array(constraints_mat * np.mat(PEFs).reshape((self.n_outcomes*self.n_settings_choice, 1))).flatten(),
            'jac': lambda x: constraints_mat
            }
        f_max = np.amax(convex_constraints['fun'](PEFs))
        return  f_max

#######################################################################################
def calc_PEFs(frequencies, beta, 
                epsilon_bias=0, delta=0, 
                default_setting=0):
    '''
    Function to calculate the PEFs and gain. Creates and object to make the computation 
    easier. The frequencies are a (4x4) 2D array with the following row/column format

    Setting    00, 01, 10, 11
    ab
    ab'
    a'b
    a'b'  
    
    The PEF calculator expects the frequencies to be ordered differently, so we must
    first take the transpose. Before returning the PEFs, we must take their transpose
    so the ordering matches the input frequencies.

    epsilon_bias is the amount of deviation in the settings from 0.5
    delta is a parameter that only applies to guarding against quantum side channels.
    default_setting is either 0 or 1.
    '''
    freq = np.array(frequencies).T
    shp = np.shape(freq)
    calc = PEF_Calculator(freq, beta, epsilon_bias=epsilon_bias, 
                            delta=delta, default_setting=default_setting)

    pefs = calc.pefs 
    pefs = pefs.reshape(shp).T
    gain = calc.gain 

    return pefs, gain


def accumulate_entropy(freq, pefs, beta, nbits_threshold, delta=0):
    """
    nbits_threshold is the smoothness error in terms of number of bits
    Utilizes the following formula to calculate the entropy accumulated given a matrix of frequencies and current pefs
    sum(sum(freq*log2(pefs/(1+ delta))))/beta
    This formula is also in the MATLAB Analysis Code line 29
    of prep_randomness_analysis.m
    """
    shape = np.shape(freq)
    N = shape[0]*shape[1]
    freq = freq.reshape(N)
    pefs = pefs.reshape(N)
    log_pefs = np.log2(pefs/(1+(delta)))
    first_term = np.sum(freq*log_pefs)
    smoothness = np.log2(2./nbits_threshold**2)
    entropy = (first_term-smoothness)/beta
    return entropy

# Extractor and experiment property calculations
def compute_minimum_trials(nbits_threshold, beta, gain, error_smoothness, isQuantum=False):
    '''
    function to compute the minimum number of trials. Taken from 
    https://arxiv.org/abs/1812.07786 equation S9
    '''
    if isQuantum:
        err = error_smoothness**2/2
    else:
        err = error_smoothness

    gamma = gain*beta
    n_exp = (nbits_threshold*beta -np.log2(err))/gamma
    n_exp = np.ceil(n_exp)
    return n_exp

def calc_threshold_bits(nbits_target,error_extractor, isQuantum=False):
    '''
    Calculate the smoothness error in terms of the number of bits (nbits_threshold)
    based on the number of target bits we desire from the extractor.
    nbits_target is the number of bits the extractor returns. It is variable k in (S7)
    nbits_input is the number of input bits to the extractor. Variable m in (S7)
    See https://arxiv.org/abs/1812.07786
    '''
    if isQuantum:
        delta_x = error_extractor**2/2
    else:
        delta_x = error_extractor

    nbits_threshold = nbits_target+4*np.log2(nbits_target)+6-4*np.log2(delta_x)
    nbits_threshold = np.ceil(nbits_threshold)

    return nbits_threshold

def calc_seed_length(nbits_threshold, nbits_input, error_extractor, isQuantum=False):
    '''
    Compute the length of the seed need for the extractor based on the smoothness error
    in terms of the number of bits, the size of the input, and the size of the extractor.
    Based on equation S7 in https://arxiv.org/abs/1812.07786
    '''
    if isQuantum:
        delta_x = error_extractor**2/2
    else:
        delta_x = error_extractor

    w = calc_w(nbits_input, nbits_threshold, delta_x)
    if w==-1:
        seed_length = -1
    else:
        e = np.exp(1)
        d_numerator = np.log2(nbits_threshold-e) - np.log2(w-e)
        d_denominator = np.log2(e) - np.log2(e-1)
        d_candidate = 1+ d_numerator/d_denominator
        seed_length = w**2 * max(2,d_candidate)
        seed_length = np.ceil(seed_length)

    return seed_length

def calc_w(m,k,delta_x):
    '''
    Parameter used in calculating the seed length. See equation S7 in
    https://arxiv.org/abs/1812.07786 equation. The inputs are
    m: the threshold entropy in bits
    k: the number of bits the extractor will return
    delta_x: related to the extractor entropy. It's value depends on 
    whether classical or quantum side information is being used.
    '''
    val = 2*np.log2(4*m*k**2/delta_x**2)
    if np.isnan(val):
        w = -1
    else:
        val = int(val)
        w = prime.next_prime(val)
    return w

##############################################################
def beta_objective(beta, params):
    if beta<=0:
        largeNum = 1E14
        return largeNum 
    freq = params['freq']
    epsilon_bias = params['epsilon_bias']
    delta = params['delta']
    nbits = params['nbits'] 
    error_smoothness = params['error_smoothness']
    error_extractor = params['error_extractor']
    # error = params['error']
    # frac_smoothness = params['frac_smoothness']
    isQuantum = params['isQuantum']

    pefs, gain = calc_PEFs(freq, beta, epsilon_bias, delta)

    # error_smoothness = frac_smoothness*error
    # error_extractor = (1-frac_smoothness)*error
    
    nbits_threshold = calc_threshold_bits(nbits, error_extractor, isQuantum=isQuantum)
    n_exp = compute_minimum_trials(nbits_threshold, beta, gain, error_smoothness, isQuantum=isQuantum)
    return n_exp[0]


# def find_optimal_beta(freq, epsilon_bias, delta, nbits, error, frac_smoothness, isQuantum):
def find_optimal_beta(freq, epsilon_bias, delta, nbits, error_smoothness, error_extractor, isQuantum):
    params = {}
    params['freq'] = freq
    params['epsilon_bias'] = epsilon_bias
    params['delta'] = delta
    params['nbits'] = nbits
    params['error_smoothness'] = error_smoothness
    params['error_extractor'] = error_extractor
    params['isQuantum'] = isQuantum

    x0 = 0.01
    res = scipy.optimize.minimize(
            beta_objective,
            x0,
            args=(params,),
            # method='SLSQP',
            tol = 1E-8,
            options={
                'disp': False,  # disabled, here for debugging purposes
                # 'ftol': tol,
                # 'maxiter': iter_limit
            })
    beta_optimal = res.x[0]
    return beta_optimal

# def find_optimal_beta(params):
#     x0 = 0.01
#     res = scipy.optimize.minimize(
#             beta_objective,
#             x0,
#             args=(params,),
#             # method='SLSQP',
#             tol = 1E-8,
#             options={
#                 'disp': False,  # disabled, here for debugging purposes
#                 # 'ftol': tol,
#                 # 'maxiter': iter_limit
#             })
#     beta_optimal = res.x[0]
#     return beta_optimal

def calculate_entropy(freq, pefs, smoothnessError, beta, epsilon_bias, delta, isQuantum=True):
    """
    smoothnessError is the smoothness error 
    Utilizes the following formula to calculate the entropy accumulated given a matrix of frequencies and current pefs
    sum(sum(freq*log2(pefs/(1+ delta))))/beta
    This formula is also in the MATLAB Analysis Code line 29
    of prep_randomness_analysis.m
        :param freq: current frequencies to use
        :param pefs: current pefs to use
        :return - the entropy
    """
    shape = np.shape(freq)
    N = shape[0]*shape[1]

    freq = freq.reshape(N)
    pefs = pefs.reshape(N)

    #qef=(sum(sum(current_analysis_freq.*log2(QEF_opt)))+log2(smoothness_error^2/2))/beta_qef_opt;
    log_pefs = np.log2(pefs/(1+(delta)))
    first_term = np.sum(freq*log_pefs)
    print('first term', first_term)
    # first_term = first_term.sum(axis=0)
    # first_term_final = first_term.sum()
    #second_term = np.log2(((4.34e-20)**2)/2.0) #Rescale for QEF
    # (L_i-log2(2/epsilon_smoothnessError^2))/beta
    if isQuantum:
        smoothness = np.log2(2./smoothnessError**2)
    else:
        smoothness = np.log2(1./smoothnessError)
    entropy = (first_term-smoothness)/beta

    return entropy


def main():
    # import zmq
    # freq = np.array([[13346400, 18186, 18982, 35939], 
    #                 [13231244, 135363, 14378, 40806], 
    #                 [13216946, 14708, 148855, 39633], 
    #                 [13056487, 172758, 184726, 3080]])

    freq = np.array([[3551853,   81632,   85159,    2437],
                   [3623020,    8211,   56850,   30070],
                   [3625515,   55481,    8894,   28782],
                   [3667943,   13706,   12605,   24827]])
    # freq = np.flipud(freq)
    # print(freq)
    beta = 0.01
    epsilon_bias = 1E-3 
    isQuantum = True 
    if isQuantum:
        delta = 4E-8
    else:
        delta = 0

    
    pefs, gain = calc_PEFs(freq, beta, epsilon_bias, delta)
    # ent1 = accumulate_entropy(freq, pefs, beta, 1089., delta=0)
    print("freq:",freq)
    print('PEFS:', pefs.reshape(4,4).tolist())
    print("LR:", str((freq/pefs.reshape(4,4)).astype(int).tolist()))
    print('gain:', gain)

    nbits = 2**23.5
    error = 2**(-64)
    frac_smoothness = 0.8
    error_smoothness = frac_smoothness*error
    error_extractor = (1-frac_smoothness)*error

    nbits_input = np.sum(freq)
    
    nbits_threshold = calc_threshold_bits(nbits, error_extractor, isQuantum=isQuantum)
    n_exp = compute_minimum_trials(nbits, beta, gain, error_smoothness, isQuantum=isQuantum)
    nbits_input = n_exp
    seed_length = calc_seed_length(nbits, nbits_input, error_extractor, isQuantum=isQuantum)

    print('length of inputs bits to extractor (m):', nbits_input)
    print('nbits threshold (k):', nbits_threshold)
    print('seed length (d):', seed_length)
    print('number of expected trials (Nexp):', n_exp)
    print('')
    print('Difference between output randomness and seed:', nbits-seed_length)
    print('Time taken:', n_exp/8E6/60)

    print('')
    print('Finding optimal beta')
    params = {}
    params['freq'] = freq
    params['epsilon_bias'] = epsilon_bias
    params['delta'] = delta
    params['nbits'] = nbits
    params['error'] = error
    params['frac_smoothness'] = frac_smoothness
    params['isQuantum'] = isQuantum
    beta_optimal = find_optimal_beta(params)
    print('optima beta:', beta_optimal)

if __name__=='__main__':
    main()
