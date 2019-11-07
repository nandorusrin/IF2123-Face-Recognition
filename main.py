import cv2
import numpy as np
import scipy
#from scipy.misc import imread
import scipy.spatial
from imageio import imread
#import cPickle as pickle
import pickle
import random
import os
import matplotlib.pyplot as plt
import math

# Feature extractor
def extract_features(image_path, vector_size=32):
    image = imread(image_path, pilmode="RGB")
    #image = imread(image_path, mode="RGB")
    try:
        # Using KAZE, cause SIFT, ORB and other was moved to additional module
        # which is adding addtional pain during install
        alg = cv2.KAZE_create()
        # Dinding image keypoints
        kps = alg.detect(image)
        # Getting first 32 of them. 
        # Number of keypoints is varies depend on image size and color pallet
        # Sorting them based on keypoint response value(bigger is better)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        # computing descriptors vector
        kps, dsc = alg.compute(image, kps)
        # Flatten all of them in one big vector - our feature vector
        dsc = dsc.flatten()
        # Making descriptor of same size
        # Descriptor vector size is 64
        needed_size = (vector_size * 64)
        if dsc.size < needed_size:
            # if we have less the 32 descriptors then just adding zeros at the
            # end of our feature vector
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
    except cv2.error as e:
        print ('Error: ', e)
        return None

    return dsc


def batch_extractor(images_path, pickled_db_path="features.pck"):
    files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]

    result = {}
    for f in files:
        print ('Extracting features from image %s' % f)
        name = f.split('/')[-1].lower()
        result[name] = extract_features(f)
    
    # saving all our feature vectors in pickled file
    with open(pickled_db_path, 'wb') as fp:
        pickle.dump(result, fp)

class Matcher(object):

    def __init__(self, pickled_db_path="features.pck"):
        #with open(pickled_db_path) as fp:
        with open(pickled_db_path,'rb') as fp:
            self.data = pickle.load(fp)
        self.names = []
        self.matrix = []
        #for k, v in self.data.iteritems():
        for k, v in self.data.items():
            self.names.append(k)
            self.matrix.append(v)
        self.matrix = np.array(self.matrix)
        self.names = np.array(self.names)

    def cos_cdist(self, vector):
        # getting cosine distance between search image and images database
        v = vector.reshape(1, -1)
        return scipy.spatial.distance.cdist(self.matrix, v, 'cosine').reshape(-1)

    def match(self, image_path, match_type, topn=5):
        features = extract_features(image_path)
        #img_distances = self.cos_cdist(features)
        if (match_type == "N"):
            img_distances = self.euclidean_distance(features)
        else:
            img_distances = self.cosine_similarity(features)
        # getting top 5 records
        nearest_ids = np.argsort(img_distances)[:topn].tolist()
        nearest_img_paths = self.names[nearest_ids].tolist()
        return nearest_img_paths, img_distances[nearest_ids].tolist()
    
    def cosine_similarity(self, vector):
        # getting cosine distance between search image and images database
        v = []
        similarity = []
        print(vector)
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
        print(vector)
        for k in vector:
            v.append(k)
        v1 = self.matrix
        v2 = v
        subxy = 0
        for i in range(len(v1)):
            for j in range(len(v1)):
                x = v1[i][j]; y = v2[j]
                subxy += (x-y)**2
                distance.append(math.sqrt(subxy))
        distance = np.array(distance)
        return distance

def show_img(path):
    img = imread(path, pilmode="RGB")
    #img = imread(path, mode="RGB")
    plt.imshow(img)
    plt.show()
    
def run():
    images_path = 'resources\images'
    files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]
    # getting 3 random images 
    M = int(input('Samples: '))
    N = int(input('Matches: '))
    match_type = str(input('Use Cosine Similarity? (Y/N): '))
    #sample = random.sample(files, 3)
    sample = random.sample(files, M)
    
    batch_extractor(images_path)

    ma = Matcher('features.pck')
    
    for s in sample:
        print ('Query image ==========================================')
        show_img(s)
        #names, match = ma.match(s, topn=3)
        names, match = ma.match(s, match_type, topn=N)
        print ('Result images ========================================')
        for i in range(N):
            #for i in range(N)
            # we got cosine distance, less cosine distance between vectors
            # more they similar, thus we subtruct it from 1 to get match value
            print ('Match %s' % (1-match[i]))
            show_img(os.path.join(names[i]))

run()