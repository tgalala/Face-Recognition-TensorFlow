#******************************************************************************
# Utilities
# This page was inspired by several websites and gihub repos
# https://github.com/ipazc/mtcnn
# https://machinelearningmastery.com/how-to-perform-face-detection-with-classical-and-deep-learning-methods-in-python-with-keras/
# https://www.pyimagesearch.com/
# https://github.com/ageitgey
# https://github.com/sthanhng
# https://www.kaggle.com/timesler/guide-to-mtcnn-in-facenet-pytorch
# https://github.com/timesler
# https://github.com/rcmalli/
# *****************************************************************************

import numpy as np
from numpy import asarray
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import os
import cv2
from scipy.cluster.vq import vq
from PIL import Image, ExifTags 
from sklearn.externals import joblib
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace

#******************************************************************************

model_dir = 'models' 


def image_exif(filename):
    '''
    Checks iphone exif information for orientation
    '''
    try:
        im=Image.open(filename)
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break
        exif=dict(im._getexif().items())

        if exif[orientation] == 6:
            im=im.rotate(270)
        return im
    
    except (AttributeError, KeyError, IndexError):
        # cases: image don't have getexif
        print('Image Detected ... ')
        print('No exif data found in this file, continuing without checking for orientation ... ')
        im=Image.open(filename)
        return im

def loadModel(modelname):
    '''
    loads a pickle with model, classLabels, standardScaler, vocabulary and size
    '''
    model, classLabels, standardScaler, vocabularySize, vocabulary = joblib.load(os.path.join(model_dir, modelname))
    return model, classLabels, standardScaler, vocabularySize, vocabulary      

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def drawBoundingBox(filename, result_list):
    data = image_exif(filename)
	# get drawing boxes
    ax = plt.gca()
    for result in result_list:
		# get coordinates
        x, y, width, height = result['box']
		# create shape
        rect = Rectangle((x, y), width, height, fill=False, color='yellow', linewidth=0.75)
        ax.add_patch(rect)
        # draw landmark keypoints
        for key, value in result['keypoints'].items():
            dot = Circle(value, radius=2, color='red')
            ax.add_patch(dot)
            

def faceDetectorCNN(filename, required_size=(224, 224)):
    '''
    Face Detection MTCNN for CNN
    '''
    image = image_exif(filename)
    pixels = asarray(image)
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    facesList = [] 
    boundingBoxes = []
    centralCoordinates = []
    if results is not None:         
        for person in results:  
            # extract bounding box from list of faces
            x1, y1, width, height = person['box']
            boundingBoxes.extend([x1,width,y1,height])
            # bug fix
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
    	    # extract the face
            x = (x1+x2)/2
            y = (y1+y2)/2
            #print(x, y)
            centralCoordinates.extend([x,y])
        	# extract the face
            face = pixels[y1:y2, x1:x2]
            image = Image.fromarray(face)
            image = image.resize(required_size)
            face_array = asarray(image, dtype='float64')
            facesList.append(face_array)
          
    return results, facesList, boundingBoxes, centralCoordinates 


def faceDetector(filename, required_size=(182, 182)):
    '''
    Face Detection MTCNN for SIFT & SURF
    '''
    image = image_exif(filename)
    pixels = asarray(image)
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    facesList = [] 
    boundingBoxes = []
    centralCoordinates = []
    if results is not None:         
        for person in results:  
            # extract bounding box from list of faces
            x1, y1, width, height = person['box']
            boundingBoxes.extend([x1,width,y1,height])
            # bug fix
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
    	    # extract the face
            x = int((x1+x2)/2)
            y = int((y1+y2)/2)
            centralCoordinates.extend([x,y])
        	# extract the face
            face = pixels[y1:y2, x1:x2]
            image = Image.fromarray(face)
            image = image.resize(required_size)
            face_array = asarray(image)
            facesList.append(face_array)        
    return results, facesList, boundingBoxes, centralCoordinates 
 
                              
def computeEmbedding(facelist,featuretype):
    '''
    Arguments:
    ***********    
    faceList: faces list from faceDetector
    featuretype: SIFT, SVM  
    Returns: descriptions array SURF = 64 columns and SIFT = 128
    ********
    '''
    # Detect, compute and return all features found on images
    if facelist is not None:     
        descriptions = []
        if featuretype == 'SURF':
            detector = cv2.xfeatures2d.SURF_create()
        elif featuretype == 'SIFT':
            detector = cv2.xfeatures2d.SIFT_create()
        else:
            raise ValueError('invalid featuretype selected, valid: SURF, SIFT')
        for face in facelist:
            grayface = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
            grayface = cv2.resize(grayface,(224,224))
            keypoints, description = detector.detectAndCompute(grayface, None)
            descriptions.append(description)
        return descriptions
    
    else:
         print('[]') 
                  
         
def bagOfFeatures(descriptions,vocabulary,vocabularySize):
    '''
    Arguments:
    ********** 
    descriptions: Output of computeEmbedding function
    vocabulary: SIFT & SURF
    vocubuarly_size: 750  
    Returns: image features
    *******
    '''
    set_size = len(descriptions)
    features = np.zeros((set_size, vocabularySize), "float32")
    for i, descriptor in enumerate(descriptions):
        words, _ = vq(descriptor, vocabulary)
        for w in words:
            features[i][w] +=1          
    return features


def reportBOF(centralCoordinates,predArray):
    '''
    Arguments:
    **********  
    centralCoordinates: central X, Y coordinates of image    
    Returns: Nx3 array where N is number of faces detected
    ********
    '''    
    centralCoordinatesArray = np.array(centralCoordinates).reshape(int(len(centralCoordinates)/2),2) # converts list to array 2xN
    return np.column_stack((predArray,centralCoordinatesArray))


def reportCNN(centralCoordinates,predicted):
    '''
    Arguments:
    **********    
    centralCoordinates: central X, Y coordinates of image    
    Returns: Nx3 array where N is number of faces detected
    ********
    '''   
    centralCoordinatesArray = np.array(centralCoordinates).reshape(int(len(centralCoordinates)/2),2) # converts list to array 2xN
    predArray = np.array(predicted)
    return np.column_stack((predArray,centralCoordinatesArray)) 