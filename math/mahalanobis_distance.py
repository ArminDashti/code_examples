'''
Mahalanobis Distance
https://jamesmccaffrey.wordpress.com/2017/11/09/example-of-calculating-the-mahalanobis-distance/
'''
import numpy as np
from numpy.linalg import inv
from scipy.linalg import sqrtm
import math

def mahalanobis_distance(x, y):
    mean = x.mean(0)
    sample_mean = y - mean
    cov = np.cov(x.T)
    inv_cov = inv(cov)
    res = (sample_mean.T @ inv_cov) @ sample_mean
    return math.sqrt(res)



x = np.array([[64,580,29],
              [66,570,33],
              [68,590,37],
              [69,660,46],
              [73,600,55]])

y = np.array([66,640,44])
mahalanobis_distance(x, y)

