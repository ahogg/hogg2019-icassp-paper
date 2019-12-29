from numpy.linalg import inv
import numpy as np

class KalmanFilter:

    def __init__(self, x, P, H, Q, R):
        self.x = x
        self.P = P
        self.H = H
        self.Q = Q
        self.R = R
        self.S = 0
        self.y = 0
        self.k = np.array([[0], [0]])


    def prediction(self, F):
        self.x = F.dot(self.x)
        self.P = F.dot(self.P).dot(F.T) + self.Q
        return

    def update(self, z):
        self.S = self.H.dot(self.P).dot(self.H.T) + self.R
        self.k = self.P.dot(self.H.T.dot(inv(self.S)))
        self.y = z - self.H.dot(self.x)
        self.x = self.x + self.k.dot(self.y)
        self.P = self.P - self.k.dot(self.H.dot(self.P))
        return

    def get_err_variance(self):
        return self.P.item(0)

    def get_err_covariance(self):
        return self.P

    def get_inno_covariance(self):
        return self.S

    def set_err_covariance(self, P):
        self.P = P
        return

    def get_kalman_gain(self):
        return self.k

    def get_state(self):
        return self.x.item(0)

    def get_post_fit_residual(self):
        return self.k.dot(self.y).item(0)
