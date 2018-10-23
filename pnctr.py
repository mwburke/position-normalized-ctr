"""
Python implementation of the following paper:
Chen, Ye and Tak W. Yan. _Position-normalized click prediction in search advertising._ KDD (2012).
https://dl.acm.org/citation.cfm?doid=2339530.2339654

Author: Matthew Burke
License: MIT
Source repo: https://github.com/mwburke/position-normalized-ctr
"""

from argparse import ArgumentParser
import numpy as np
import pandas as pd
from scipy.special import gamma


class PNCTR:

    def __init__(self, data, alpha=1, beta=5, convergence=0.01, verbose=0):
        self.data = data
        self.data['ctr'] = self.data['clicks'] / self.data['impressions']
        self.alpha = alpha
        self.beta = beta
        self.convergence = convergence
        self.verbose = verbose
        self.calculated = False
        self.steps = 0

    def initialize_p_q(self):
        """
        Set initial p values to the average CTRs for each ad_query
        and the initial q values to the average CTRs for each position.

        Overwrites any existing values upon re-run.
        """
        ad_querys_grouped = self.data[['ad_query', 'ctr']].groupby('ad_query').mean().reset_index()
        print(ad_querys_grouped)
        self.i_values = np.array(ad_querys_grouped['ad_query'])  # .tolist()
        # self.p = np.array(ad_querys_grouped['ctr'].tolist())
        self.p = np.random.rand(len(self.i_values)) * np.mean(self.data['ctr'])

        positions_grouped = self.data[['position', 'ctr']].groupby('position').mean().reset_index()
        print(positions_grouped)
        self.j_values = np.array(positions_grouped['position'])  # .tolist()
        # self.q = np.array(positions_grouped['ctr'].tolist())
        self.q = np.random.rand(len(self.j_values)) * np.mean(self.data['ctr'])

        self.p_prev = np.ones(len(self.p))
        self.q_prev = np.ones(len(self.q))
        self.calculated = False

        if self.verbose == 1:
            print('Initialized')

    def update_qj(self, j):
        """
        Calculates estimated CTR for a single position value (q_j)
        """
        data = self.data[self.data['position'] == j]
        numerator = np.sum(data['clicks']) + (self.alpha - 1) * data.shape[0]
        divisor = np.sum([data[data['ad_query'] == i]['impressions'].values[0] * self.p[i_ind] + (1 / self.beta) for i_ind, i in enumerate(self.i_values)])
        return numerator / divisor

    def update_pi(self, i):
        """
        Calculates optimal CTR for a single ad_query value (p_i)
        """
        data = self.data[self.data['ad_query'] == i]
        return np.sum(data['clicks'] / (data['impressions'] * data['position']))

    def perform_em_step(self):
        """
        Performs single step of the EM procedure:
            Runs estimation step to update position parameters
            and then runs maximization step to find the optimal
            ad_query parameters that optimize those paramters.
        """

        # E step
        if self.verbose == 1:
            print('Starting E step')
        self.q = np.array([self.update_qj(j) for j in self.j_values])
        # M step
        if self.verbose == 1:
            print('Starting M step')
        self.p = np.array([self.update_pi(i) for i in self.i_values])

    def estimate(self):
        """
        Sets the initial parameter estimates and continually performs the
        EM steps until the change in the estimated parameters changes less
        than the convergence value between steps.
        """
        self.initialize_p_q()
        self.log_likelihoods = []
        while (np.abs(np.mean(self.q - self.q_prev)) > self.convergence) & \
              (np.abs(np.mean(self.p - self.p_prev)) > self.convergence):

            self.steps += 1

            self.p_prev = self.p
            self.q_prev = self.q
            if self.verbose == 1:
                print('Starting EM step')
            self.perform_em_step()
            if self.verbose == 1:
                print('Starting log likelihood calculation')
            log_likelihood = self.log_likelihood()
            self.log_likelihoods.append(log_likelihood)

            if self.verbose == 1:
                print('Completed EM Step', self.steps)
                print('Log Likelihood:', np.round(log_likelihood, 3))

        self.calculated = True

        if self.verbose == 1:
            print('Completed estimation')

    def log_likelihood(self):
        """
        Calculate the log likelihood of the current set of p and q values
        including the prior values for q
        """
        data_likelihood = 0
        prior_likelihood = 0
        for j_ind, j in enumerate(self.j_values):
            prior_likelihood += (self.alpha - 1)\
                * np.log(self.q[j_ind])\
                - self.q[j_ind] / self.beta\
                - self.alpha * self.beta\
                - np.log(gamma(self.alpha))

            for i_ind, i in enumerate(self.i_values):
                data = self.data[(self.data['ad_query'] == i) & (self.data['position'] == j)]
                data_likelihood = data['clicks'] * \
                    np.log(data['impressions'] *
                           self.p[i_ind] *
                           self.q[j_ind]) \
                    - data['impressions'] \
                    * self.p[i_ind] \
                    * self.q[j_ind] \
                    - np.log(np.math.factorial(data['clicks']))

        return data_likelihood + prior_likelihood

    def get_p_values(self):
        if self.calculated:
            return {i: value for i, value in zip(self.i_values, self.p)}
        else:
            print('Please run estimation first')

    def get_q_values(self):
        if self.calculated:
            return {j: value for j, value in zip(self.j_values, self.q)}
        else:
            print('Please run estimation first')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", dest="filename",
                        help="input csv file")
    parser.add_argument("-a", "--alpha", dest="alpha", type=float, default=1,
                        help="input alpha parameter")
    parser.add_argument("-b", "--beta", dest="beta", type=float, default=0.05,
                        help="input beta parameter")
    parser.add_argument("-c", "--convergence", dest="convergence", type=float, default=0.01,
                        help="input convergence tolerance limit")
    parser.add_argument("-v", "--verbosity", dest="verbosity", type=int, default=0,
                        help="set to 1 for all print updates")

    args = parser.parse_args()

    data = pd.read_csv(args.filename)

    pnctr = PNCTR(data, args.alpha, args.beta, args.convergence, args.verbosity)
    pnctr.estimate()

    print('Ad-Query CTRs:')
    print(pnctr.get_p_values())
    print()
    print('Position prior CTRs:')
    print(pnctr.get_q_values())
