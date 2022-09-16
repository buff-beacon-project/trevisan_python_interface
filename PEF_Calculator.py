import os
import time
import itertools
import functools
import math
import numpy as np
import scipy.optimize
import scipy.interpolate
from scipy.optimize import fsolve
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from PEF_Analysis import PEF_Analysis
import cvxpy as cvx

epsilon = np.finfo(np.float64).eps
np.set_printoptions(precision=128, threshold=10**15)

class PEF_Calculator(PEF_Analysis):
    """At the beginning, this class is to be instantiated to calculate the probability estimation factors.
    """
    def __init__(self, frequencies, beta, epsilon_bias, delta, min_trials= 4, **kwargs):
        super().__init__(beta,  epsilon_bias, delta, **kwargs)
        self.min_trials = min_trials
        self.pefs = np.zeros((16,))

        # Find conditional frequencies consistent with non-signaling.
        self.conditional_freqs = self.compute_ns_conditional_frequencies(frequencies)

        # Account for violated CHSH inequality (if one is violated)
        I_CHSH = self._find_opt_I_CHSH(self.conditional_freqs)
        V = np.longdouble(np.sqrt(2) - 1) #why is the violation sqrt(2) -1 and not sqrt(2) -2?
        A = self.mLR[np.array(self.mLR * I_CHSH.transpose(), dtype  = np.longdouble).flatten() == 2, :]
        B = self.mPR[np.array(self.mPR * I_CHSH.transpose(), dtype  = np.longdouble).flatten() == 4, :]
        self.C = (1 - V) * A + V * np.tile(B, (len(A), 1))

        # This is set of constraints in addition to non-negativity is the model assumed. Note that it is strictly bigger than Q and strictly smaller than NS.
        self.ns_vectors = np.concatenate((
            self.mLR,
            self.C,
            self.mPR[np.array(self.mPR * I_CHSH.transpose(), dtype  = np.longdouble).flatten() != 4, :]))


    def get_current_pefs(self):
        """ @author: Aliza Siddiqui
         Returns the current pefs calculated. Mainly for outside classes if they need to access the class variable, self.pefs
            :return - the current pefs
        """
        return self.pefs



    def compute_ns_conditional_frequencies(self, frequencies):
        """ @author: Mohammad Alhejji
        Computes the best conditional frequencies consistent with non-signaling constraints
            :param frequencies - The raw frequencies
            :return - The conditional frequencies
        """
        freq = frequencies.reshape(16) #freq is now a 1x16
        freq = freq/np.sum(freq)  #Normalize

        cond_freq_mat = cvx.Variable((4, 4))
        cond_freq_var = cvx.reshape(cond_freq_mat.T, (16, 1)) #16x1

        obj = cvx.Minimize(-freq * cvx.log(cond_freq_var))

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
        a_ns_constraints = [outcome_a * cond_freq_mat[:, bob_settings[0]] == outcome_a * cond_freq_mat[:, setting_b]
                            for setting_b in bob_settings[1:] for outcome_a in alice_outcomes]
        # Bob's outcome should be independent of Alice's setting
        b_ns_constraints = [outcome_b * cond_freq_mat[:, alice_settings[0]] == outcome_b * cond_freq_mat[:, setting_a]
                            for setting_a in alice_settings[1:] for outcome_b in bob_outcomes]

        # Tsirelson bound + NS conditions
        I_CHSH_all = self.mCHSH
        tsirelson_bound_constraints = [
            I_CHSH_all * cond_freq_var <= 2*np.sqrt(2) # There are 8 variants because CHSH is between -2 and 2.
        ]
        constraints = [cond_freq_mat >= 0,
                      cvx.sum(cond_freq_mat, axis=0) == 1,
        ] + a_ns_constraints + b_ns_constraints + tsirelson_bound_constraints

        prob = cvx.Problem(obj, constraints)
        prob.solve('ECOS', abstol=1e-15)
        cond_freq = np.array(cond_freq_mat.value).reshape(16)
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
        print("CHSH Models to be Considered", np.mat(I_CHSH))
        return np.mat(I_CHSH)


    def compute_LR_stat_strength(self, frequencies, settings_weight = np.array([1/4,1/4,1/4,1/4], dtype = np.longdouble)):
        """ @author: Mohammad Alhejji
        Computes the statistical strength of given frequencies with respect to the KL divergence and the LR polytope.
            :param frequencies - The raw frequencies
            :param settings_weight - The distribution on the input settings
            :return - The statistical strength
        """
        freq = frequencies.reshape(4,4)
        freq = self.compute_ns_conditional_frequencies(freq).reshape(4,4)
        joint_freq = (freq * np.tile(settings_weight, (self.n_outcomes, 1))).reshape(16)
        freq = freq.reshape(16)
        cond_freq_mat = cvx.Variable((4, 4))
        cond_freq_var = cvx.reshape(cond_freq_mat.T, (16, 1))

        obj = cvx.Minimize(-joint_freq * cvx.log(cond_freq_var) + joint_freq*cvx.log(freq))

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
        a_ns_constraints = [outcome_a * cond_freq_mat[:, bob_settings[0]] == outcome_a * cond_freq_mat[:, setting_b]
                            for setting_b in bob_settings[1:] for outcome_a in alice_outcomes]
        # Bob's outcome should be independent of Alice's setting
        b_ns_constraints = [outcome_b * cond_freq_mat[:, alice_settings[0]] == outcome_b * cond_freq_mat[:, setting_a]
                            for setting_a in alice_settings[1:] for outcome_b in bob_outcomes]

        # CHSH bound + NS conditions
        I_CHSH_all = self.mCHSH
        CHSH_bound_constraints = [
            I_CHSH_all * cond_freq_var <= 2 # There are 8 variants in:sent because CHSH is between -2 and 2.
        ]
        constraints = [cond_freq_mat >= 0,
                      cvx.sum(cond_freq_mat, axis=0) == 1,
        ] + a_ns_constraints + b_ns_constraints + CHSH_bound_constraints
        prob = cvx.Problem(obj, constraints)
        prob.solve('ECOS', abstol = 1e-15, feastol = 1e-12)
        print("\n\nLocal Realism Statistical Strength: ", prob.value/np.log(2))
        return prob.value/np.log(2)


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
        settings_bias /= sum(settings_bias)

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

        settings_bias1 = np.array([p**2, p*q, p*q, q**2], dtype= np.longdouble)
        settings_bias2 = np.array([p*q, q**2, p**2, p*q], dtype= np.longdouble)
        settings_bias3 = np.array([p*q, p**2, q**2, p*q], dtype=np.longdouble)
        settings_bias4 = np.array([q**2, p*q, p*q, p**2], dtype=np.longdouble)

        freq = self.conditional_freqs.reshape((self.n_outcomes, self.n_settings_choice)) * settings_bias1
        freq = freq.reshape(num_vars)
        freq = freq/np.sum(freq)  # Normalize


        #Calculating the convex constraints for each probability distribution
        convex_constraints1 = self.calculate_convex_constraints(settings_bias1, beta, num_vars)
        convex_constraints2 = self.calculate_convex_constraints(settings_bias2, beta, num_vars)
        convex_constraints3 = self.calculate_convex_constraints(settings_bias3, beta, num_vars)
        convex_constraints4 = self.calculate_convex_constraints(settings_bias4, beta, num_vars)


        return convex_constraints1, convex_constraints2, convex_constraints3, convex_constraints4, freq

    def optimizing_PEFS(self, objf, x, jac, constraints, tol, iter_limit):
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
            options={
                'disp': False,  # disabled, here for debugging purposes
                'ftol': tol,
                'maxiter': iter_limit,
            })
        return res

    def get_pefs(self,pos_results, convex_constraints, epsilon):
        """ HELPER FUNCTION TO _calc_PEF_and_gain
            @author: Aliza Siddiqui
        Makes extra sure that the constraints are satisfied and gives the PEFS
            :param pos_results - The objective function
            :param convex constraints - the convex constraints for a certain probability distribution
            :param epsilon -
            :return - The PEFs
        """
        correction_vals = [] #list of correction values
        pefs = [] #list of pefs

        for pos_result in pos_results:
            correction_vals.append(1-min(convex_constraints['fun'](pos_result)) + epsilon)

        for i in range(4):
            pefs.append(pos_results[i]/correction_vals[i])
        return pefs

    def _calc_PEF_and_gain(
            self,
            beta,
            epsilon_bias,
            guess=None,
            iter_limit=10000,
            default_setting=0,
            tol=1e-18
            ):
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
            num_vars = self.n_outcomes * self.n_settings_choice
            convex_constraints1, convex_constraints2, convex_constraints3, convex_constraints4, freq = self.place_convex_constraints(epsilon_bias, beta, num_vars)


        # Objective function and gradient
            objf = lambda x: self.log_likelihood(x, freq)
            jac = lambda x: self.log_likelihood_grd(x, freq)


        # PEFs are positive
            positive_constraints = {
            'type': 'ineq',
            'fun': lambda x: x,
            'jac': lambda x: np.mat(np.eye(num_vars))
            }

        # PEFs for no-click no-click are set to 1 for the default setting
            ind00 = [default_setting]
            trivial_pef_constraints = [{
                  'type': 'eq',
                  'fun': functools.partial(lambda idx, x: x[idx] - 1, ind),
                  'jac': functools.partial(
                   lambda idx, x: np.hstack((np.zeros((1, idx)), np.eye(1), np.zeros((1, num_vars-1-idx)))),
                   ind)
                   } for ind in ind00]

        # starting values for SLSQP

            x0 = guess if guess is not None else np.array([1,1/3,1/100,1/2,1/200,1/21,1/24,1/25000,1/200,1/113,1/2000,1/2000,1/234,1/201,1/200,1/2000], dtype = np.longdouble)
            x1 = guess if guess is not None else np.array([1,1/2,1/2,1/2,1/2,1/2,1/2,1/2,1/2,1/2,1/2,1/2,1/2,1/2,1/2,1/2], dtype = np.longdouble)
            x2 = guess if guess is not None else np.array([1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1], dtype = np.longdouble)
            x3 = guess if guess is not None else np.array([0.9999999999999996, 1.0000322126232337, 1.00003263372623  , 0.9999351536905512,
            0.9999999490254485, 0.9960292568771285,
            0.9490167069215524, 1.0039381094066182, 0.999999952393381 ,
            0.947250529326817 , 0.9957175447985281, 1.0042502426132405,
            1.0000000496990962, 1.0209886928359673, 1.0224427372692761,
            0.9305120790184543], dtype = np.longdouble)

        # optimize
            constraints = [convex_constraints1, convex_constraints2, convex_constraints3, convex_constraints4, positive_constraints] + trivial_pef_constraints

            res = self.optimizing_PEFS(objf,  x0, jac, constraints, tol, iter_limit)
            res1 = self.optimizing_PEFS(objf, x1, jac, constraints, tol, iter_limit)
            res2 = self.optimizing_PEFS(objf, x2, jac, constraints, tol, iter_limit)
            res3 = self.optimizing_PEFS(objf, x3, jac, constraints, tol, iter_limit)

            #print("res: ", res)

            #self.compute_minimum_trials(res, res1, res2, res3)

        # Make extra sure that the constraints are satisfied

            pos_result_0 = res.x
            pos_result_1 = res1.x
            pos_result_2 = res2.x
            pos_result_3 = res3.x
            pos_results = [pos_result_0, pos_result_1, pos_result_2, pos_result_3]

            pos_result_0[pos_result_0 < 0] = 0
            pos_result_1[pos_result_1 < 0] = 0
            pos_result_2[pos_result_2 < 0] = 0
            pos_result_3[pos_result_3 < 0] = 0

            PEFs = np.concatenate((self.get_pefs(pos_results, convex_constraints1, epsilon),
                                   self.get_pefs(pos_results, convex_constraints2, epsilon),
                                   self.get_pefs(pos_results, convex_constraints3, epsilon),
                                   self.get_pefs(pos_results, convex_constraints4, epsilon)))

            # Expected Gain (Entropy)
            gains = np.array([-objf(PEFs[i])/beta for i in range(0,4)])
            gain = np.amax(gains); gain_index = np.argmax(gains)
            PEF = PEFs[gain_index]
            self.pefs = PEF
            print("PEFS: ", PEF)
            print("Expected Gain: ", gain)
            return PEF, gain #returns 16x1 array of the optimal PEFs as well as a decimal value of the gain



    '''
    def compute_minimum_trials(self, gain):
        #TODO: Not sure if this is right
        n = (1089 + (math.log2(2.0/(4.34e-20)**2))/0.01)/gain
        return 2*n
    '''

    def get_minimum_trials(self):
        return self.min_trials


    @classmethod
    def log_likelihood(cls, x, freq):
        """
          @author: Mohammad Alhejji
        Computes the log likelihood of x
        Solves the following: Max{ E_v(log_2(PEF)) }
          :param x:
          :param freq:
          :return: log likelihood
        """
        x = x[freq > 0]
        if min(x) < 1e-12:
            return np.inf
        freq = freq[freq > 0]
        return - freq.dot(np.log2(x, dtype = np.longdouble))

    @classmethod
    def log_likelihood_grd(cls, x, freq):
        """
          @author: Mohammad Alhejji
        Computes the gradient of the log likelihood (the fastest rate at which log likelihood is increasing)
          :param x:
          :param freq:
          :return:  gradient
        """
        gradient = np.zeros(len(freq))
        gradient[freq > 0] = - np.array(freq[freq > 0]) * 1./(np.array(x[freq > 0]) * np.log(2))
        return gradient


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

        if sum(settings_weight) > 1 + epsilon:
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


def main():
    import zmq

    calc = PEF_Calculator(freq.T, beta, epsilon_bias, delta)
