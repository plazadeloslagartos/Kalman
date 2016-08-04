"""
This scripts serves as a demonstration of Kalman Filter tracking of brownian motion data
"""

import scipy as sp
from scipy.linalg import inv
from functools import wraps


_t_delta = 0.5

_A = sp.array([[1, _t_delta], [0, 1]])
_H = sp.array([[1, 0], [0, 1]])
_R = None
_Q = sp.array([[0.001, 0], [0, .001]])


class KalmanFilter(object):
    """Class for implementing simple Kalman Filter, assumes no Control input"""
    def __init__(self, A=_A, H=_H, R=_R, Q = _Q):
        dim = A.shape[0]
        self.A = A  # Transition matrix
        self.H = H  # Extraction matrix
        self.R = R  # Covariance matrix, measurement noise
        self.Q = Q  # Covariance matrix, process noise
        self.x_mu_prior = sp.zeros([dim, 1])
        self.x_mu = sp.zeros([dim, 1])
        self.P_prior = sp.zeros([dim, dim])
        self.P = sp.zeros([dim, dim])
        self.P[-1][-1] = .001
        self.I = sp.identity(dim)

    def set_R(self, R_new):
        self.R = R_new

    def check_R(func):
        @wraps(func)
        def check_dec(self, *args):
            if self.R is not None:
                return func(self, *args)
            else:
                raise ValueError("R has not been initialized!")
        return check_dec

    @check_R
    def predict(self):
        A = self.A
        Q = self.Q
        x_mu = self.x_mu
        P = self.P
        self.x_mu_prior = A.dot(x_mu)
        self.P_prior = A.dot(P.dot(A.T)) + Q

    @check_R
    def calculate_kalman_gain(self):
        # Improve inverse using cholesky factorization
        H = self.H
        P_prior = self.P_prior
        R = self.R
        S = H.dot(P_prior.dot(H.T)) + R
        K = P_prior.dot(H.T.dot(inv(S)))
        return K

    @check_R
    def update(self, z_new):
        H = self.H
        x_mu_prior = self.x_mu_prior
        P_prior = self.P_prior
        K = self.calculate_kalman_gain()
        I = self.I
        self.x_mu = x_mu_prior + K.dot(z_new - H.dot(x_mu_prior))
        self.P = (I - K.dot(H)).dot(P_prior)
