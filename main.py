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
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

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


def batch_extractor(files, pickled_db_path="features.pck"):
    files = files

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
    
class App(QWidget):
    def __init__(self):
        super().__init__()
        
        self.title = "Face Recognition App"
        self.window_width = 640
        self.window_height = 480
        self.files = []
        self.res = []
        self.match_type = 'Y'
        self.setFixedSize(self.window_width, self.window_height)
        self.initUI()

    def createLayout_group(self, lst, number):
        sgroupbox = QGroupBox("Image {}:".format(number + 1), self)
        layout_groupbox = QVBoxLayout(sgroupbox)
        for l in lst:
            item = QLabel()
            item_lst = QPixmap(QCoreApplication.applicationDirPath() + lst)
            item_lst = item_lst.scaled(50, 50, Qt.KeepAspectRatio, Qt.SmoothTransformation);
            layout_groupbox.addWidget(item.setPixmap(item_lst))
        layout_groupbox.addStretch(1)
        return sgroupbox

    def createLayout_Container(self, lst):
        self.scrollarea = QScrollArea(self)
        self.scrollarea.setFixedHeight(0.8 * self.window_height)
        self.scrollarea.setWidgetResizable(True)

        widget = QWidget()
        self.scrollarea.setWidget(widget)
        self.layout_SArea = QVBoxLayout(widget)

        for i in range(len(lst)):
            self.layout_SArea.addWidget(self.createLayout_group(lst[i], i))
        self.layout_SArea.addStretch(1)

    def initUI(self):
        self.grid = QGridLayout()
        self.grid.setSpacing(10)
        
        self.btn = QPushButton(self)
        self.btn.setText('Select image')
        self.btn.clicked.connect(self.getImage)

        self.btn2 = QPushButton(self)
        self.btn2.setText('Compare')
        self.btn2.clicked.connect(self.run)
        
        self.createLayout_Container(self.files)
        self.grid.addWidget(self.scrollarea, 1, 0)
        
        self.createLayout_Container(self.res)
        self.grid.addWidget(self.scrollarea, 1, 1)

        self.grid.addWidget(self.btn, 2, 0)

        self.grid.addWidget(self.btn2, 2, 1)
        
        self.setLayout(self.grid) 
        self.show()
        
        self.initEngine()
    
    def initEngine(self):
        images_path = 'resources\images'
        pins_dir = 'resources\pins-face-recognition'

        files_arr = []
        i = 0
        limit = 20

        for subdir, dirs, files in os.walk(pins_dir):
            for file in files:
                files_arr.append(os.path.join(subdir, file))
                i =  i + 1
                if i > limit:
                    break
            if i > limit:
                break

        batch_extractor(files_arr)
    
    def getImage(self):            
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(self, 'Open file',
                                            "", "Image files (*.jpg *.gif)", options=options)
        
        if files:
            self.files = []
            
            for f in files:
                self.files.append(f)

            self.createLayout_Container(self.files)
            self.grid.addWidget(self.scrollarea, 1, 0)
    
    def run(self):
        ret = QMessageBox.question(self,'', "Use Cosine similarity?", QMessageBox.Yes | QMessageBox.No)
        
        if ret == QMessageBox.Yes:
            self.match_type = 'Y'
        else:
            self.match_type = 'N'
            
        ma = Matcher('features.pck')
        
        if not self.files:
            print ('No files selected')
        else:
            self.res = []

            for s in self.files:
                print ('Query image ==========================================')
                show_img(s)
                names, match = ma.match(s, self.match_type, topn=3)
                print ('Result images ========================================')
                for i in range(3):
                    # we got cosine distance, less cosine distance between vectors
                    # more they similar, thus we subtruct it from 1 to get match value
                    print ('Match %s' % (1-match[i]))
                    img = os.path.join(names[i])
                    show_img(img)
                    
                    self.res.append(img)

            self.createLayout_Container(self.res)
            self.grid.addWidget(self.scrollarea, 1, 1)
            
            self.files = []

            
def main():
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
    
if __name__ == '__main__':
    main()