import cv2
import numpy as np
import scipy
from scipy.misc import imread
import cPickle as pickle
import random
import os
import matplotlib.pyplot as plt
import math

def cosine_similarity(self, vector):
        # getting cosine distance between search image and images database
        v = []
        similarity = []
        print vector
        for k in vector:
            v.append(k)
        v1 = self.matrix
        v2 = v
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(v1)):
            for j in range(len(v1)):
                x = v1[i][j]; y = v2[j]
                sumxx += x*x
                sumyy += y*y
                sumxy += x*y
                similarity.append(sumxy/math.sqrt(sumxx)*math.sqrt(sumyy))
        similarity = np.array(similarity)
        return similarity

    def euclidean_distance(self, vector):
        # getting cosine distance between search image and images database
        v = []
        distance = []
        print vector
        for k in vector:
            v.append(k)
        v1 = self.matrix
        v2 = v
        subxy= 0
        for i in range(len(v1)):
            for j in range(len(v1)):
                x = v1[i][j]; y = v2[j]
                subxy += (x-y)**2
                distance.append(math.sqrt(subxy))
        distance = np.array(distance)
        return distance