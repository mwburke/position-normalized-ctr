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
        Set initial p values to the average CTRs for each ad
        and the initial q values to the average CTRs for each position.

        Overwrites any existing values upon re-run.
        """
        ads_grouped = self.data[['ad_query', 'ctr']].groupby('ad_query').mean().reset_index()
        self.i_values = np.array(ads_grouped['ad_query'])  # .tolist()
        self.p = ads_grouped['ctr'].tolist()

        positions_grouped = self.data[['position', 'ctr']].groupby('position').mean().reset_index()
        self.j_values = np.array(positions_grouped['position'])  # .tolist()
        self.q = positions_grouped['ctr'].tolist()

        self.calculated = False

        if self.verbose == 1:
            print('Initialized')

    def perform_em_step(self):
        """
        Performs single step of the EM procedure:
            Runs estimation step to update position parameters
            and then runs maximization step to find the optimal
            ad-query parameters that optimize those paramters.
        """

        def update_qj(self, j):
            """
            Calculates estimated CTR for a single position values (q_j)
            """
            data = self.data[self.data['position'] == j]
            numerator = np.sum(data['clicks']) + (self.alpha - 1) * data.shape[0]
            divisor = np.sum(data['views'] * data['ad_query']) + (1 / self.beta) * data.shape[0]
            return numerator / divisor

        def update_pi(self, i):
            """
            Calculates optimal CTR for a single ad-query value (p_i)
            """
            data = self.data[self.data['ad_query'] == i]
            return np.sum(data['clicks'] / (data['views'] * data['position']))

        # E step
        self.q = [self.update_qj(j) for j in self.j_values]
        # M step
        self.p = [self.update_pi(i) for i in self.i_values]

    def estimate(self):
        """
        Sets the initial parameter estimates and continually performs the
        EM steps until the change in the estimated parameters changes less
        than the convergence value between steps.
        """
        self.initialize_p_q()
        self.log_likelihoods = []
        while (np.abs(np.mean(self.q - self.q_prev)) >= self.convergence) & \
              (np.abs(np.mean(self.p - self.p_prev)) >= self.convergence):

            self.step += 1

            self.p_prev = self.p
            self.q_prev = self.q
            self.perform_em_step()
            log_likelihood = self.log_likelihood()
            self.log_likelihoods.append(log_likelihood)

            if self.verbose == 1:
                print('Completed EM Step', self.step)
                print('Log Likelihood:', np.round(log_likelihood, 3))

        self.calculated = True

    def log_likelihood(self):
        """
        Calculate the log likelihood of the current set of p and q values
        including the prior values for q
        """
        data_likelihood = 0
        prior_likelihood = 0
        for j in self.j_values:
            prior_likelihood += (self.alpha - 1)\
                * np.log(self.q[np.where(self.q_values == j)])\
                - self.q[np.where(self.q_values == j)] / self.beta\
                - self.alpha * self.beta\
                - np.log(gamma(self.alpha))

            for i in self.i_values:
                data = self.data[(self.data['ad_query'] == i) & (self.data['position'] == j)]
                data_likelihood = data['clicks'] * \
                    np.log(data['impressions'] *
                           self.p[np.where(self.p_values == i)] *
                           self.q[np.where(self.q_values == j)]) \
                    - data['impressions'] \
                    * self.p[np.where(self.p_values == i)] \
                    * self.q[np.where(self.q_values == j)] \
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
    parser.add_argument("-a", "--alpha", dest="alpha", default=1,
                        help="input alpha parameter")
    parser.add_argument("-b", "--beta", dest="beta", default=5,
                        help="input beta parameter")
    parser.add_argument("-c", "--convergence", dest="convergence", default=0.01,
                        help="input convergence tolerance limit")
    parser.add_argument("-v", "--verbose", dest="verbose", default=0,
                        help="set to 1 for all print updates")

    args = parser.parse_args()

    data = pd.read_csv(args['filename'])

    pnctr = PNCTR(data, args['alpha'], args['beta'], args['convergence'])
    pnctr.estimate()

    print('Ad-query CTRs:')
    print(pnctr.get_p_values)
    print()
    print('Position prior CTRs:')
    print(pnctr.get_q_values)
