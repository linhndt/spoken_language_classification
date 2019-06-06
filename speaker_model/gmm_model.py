from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, MiniBatchKMeans
import scipy.stats as stats
import numpy as np


class GaussianMixtureModel:

    def __init__(self, num_of_components=32):
        self.num_of_gaussian_component = num_of_components

        # According to research, GMM in speaker recognition
        # work best with diagonal covariance matrix
        self.gmm_model = GaussianMixture(n_components=num_of_components,
                                         covariance_type="diag")

    def fit(self, data):
        # Use K-Mean to estimate GMM component centers
        # kmeans = KMeans(n_clusters=self.num_of_gaussian_component)
        kmeans = MiniBatchKMeans(n_clusters=self.num_of_gaussian_component)
        kmeans.fit(data)

        self.gmm_model.means_init = kmeans.cluster_centers_

        # Fit data
        self.gmm_model.fit(data)

    def adapt(self, data):
        """
        Adapting Gaussian components to a specific speaker's data. Use for UBM model

        Ref: Speaker Verification Using Adapted Gaussian Mixture Models - D. A. Reynolds et al.
        """

        # NOTE: Self-convention: L stand for list, D stand for dict, w stand for weighted
        # Key: i - index of component; Value: probability density of each data point with component i
        print("Start adapting GMM to data size %s", str(data.shape))
        T = len(data)                       # number of data point
        M = len(self.gmm_model.means_)      # number of Gaussian components

        print("Start calculating pdf of %d components with %d data points", M, T)

        pD = np.transpose(self.gmm_model.predict_proba(data))     # Shape: n_components, n_sameple


        # pD = dict()
        # for i in range(M):
        #     mu = self.gmm_model.means_[i]
        #     cov = self.gmm_model.covariances_[i]
        #
        #     pD[i] = list()
        #
        #     for xt in data:
        #         # pD[i][t] = gauss_pdf of component i with data point xt
        #         pxt = stats.multivariate_normal.pdf(xt, mu, cov)
        #         pD[i].append(pxt)
        #
        #     logger.debug("Finish component %d", i)

        # Weighted probability density of each data point
        # wpL[t] = pdf of GMM with data point xt

        print("Start calculating weight pdf of %d data points", T)
        wpL = list()
        for t, xt in enumerate(data):

            prob_sum = 0
            for i in range(M):
                prob_sum += pD[i][t] * self.gmm_model.weights_[i]

            wpL.append(prob_sum)

        # Normalized pdf of component i with data point xt
        # Correspond with formula 7 in the article
        # prD[i][t]= (w_i * p_i(t) / sum (w_j * p_j(t)))
        prD = dict()

        print("Start normalize weight pdf of %d components with %d data points", M, T)
        for i in range(M):
            wi = self.gmm_model.weights_[i]
            prD[i] = list()

            for t, xt in enumerate(data):
                prD[i].append(pD[i][t] * wi / wpL[t])

        nL = dict()
        eL = dict()
        e2L = dict()
        wL_ = list()

        for i in range(M):
            print("Start adapting parameters of GMM component %d", i)
            mui = self.gmm_model.means_[i]
            cov = np.diag(self.gmm_model.covariances_[i])  # because GMM store covariance matrix as diagonal vector
            wi = self.gmm_model.weights_[i]

            nL[i] = sum(prD[i])
            # in case the total sum of probability too small
            # skip adapting this component
            if nL[i] == 0:
                logger.warn("Total sum of probability = 0. Skip this component")
                wL_.append(wi)
                continue

            eL[i] = 1 / nL[i] * np.sum([pD[i][t] * data[t] for t in range(T)], axis=0)
            e2L[i] = (1 / nL[i] * np.sum([pD[i][t] * self.square(data[t]) for t in range(T)], axis=0))

            # Use one adaption cofficient for weight, mean and covariance
            r = 16
            alpha = nL[i] / (nL[i] + r)

            wi_ = alpha * nL[i] / T + (1 - alpha) * wi

            mui_ = alpha * eL[i] + (1 - alpha) * mui

            # Disable covariance adapting since new covariance has negative element in diagonal
            # cov_ = alpha * e2L[i] + (1 - alpha) * (cov + self.square(mui)) - self.square(mui_)

            print("Old mean", mui)
            print("New mean", mui_)
            print("Old weight", wi)
            print("New weight", wi_)

            self.gmm_model.means_[i] = mui_
            # self.gmm_model.covariances_[i] = np.diag(cov_)      # only take diagonal matrix

            wL_.append(wi_)

        total_w = sum(wL_)

        for i in range(M):
            # Normalize weight to ensure total weight sum = 1
            self.gmm_model.weights_[i] = wL_[i] / total_w

    def square(self, vector):
        return np.outer(vector, vector)

    def log_proba(self, data):
        return self.gmm_model.score(data)
