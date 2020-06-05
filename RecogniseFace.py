#******************************************************************************
from utilities import *

#******************************************************************************

import numpy as np
from numpy import asarray
import sys
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from keras.models import Model 
from keras.layers import Dense, Flatten 
from keras_vggface.vggface import VGGFace
from keras_vggface import utils

#******************************************************************************

model = VGGFace(model='vgg16')

#custom parameters
number_of_classes = 56
hidden_dim = 512

VGG16_Freeze_Top_Layers = VGGFace(include_top=False, input_shape=(224, 224, 3))

for layer in VGG16_Freeze_Top_Layers.layers:
    layer.trainable = False

last_layer = VGG16_Freeze_Top_Layers.get_layer('pool5').output
x = Flatten(name='flatten')(last_layer)
x = Dense(hidden_dim, activation='relu', name='fc6')(x)
x = Dense(hidden_dim, activation='relu', name='fc7')(x)
out = Dense(number_of_classes, activation='softmax', name='fc8/activation')(x)

VGGmodel = Model(VGG16_Freeze_Top_Layers.input, out)

# load weights into new model
VGGmodel.load_weights("models/modelVGG.h5")

# define class_names
class_names = ['1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '36', '38', '4', '40', '42', '44', '46', '48', '5', '50', '52', '54', '56', '58', '6', '60', '7', '78', '8', '9', 'unknown 1', 'unknown 2', 'unknown 3', 'unknown 4', 'unknown 5', 'unknown 6', 'unknown 7', 'unknown 8']

#******************************************************************************
def RecogniseFace(I,featureType,classifierType,creativeMode):
    '''
    Arguments
    **********
    I: Input image for face recoginition
    featureType: SURF, SIFT, NONE
    classifierType: SVM, LR, VGG16
    creativeMode: 0: none, 1: Anonymity, 2: Cool Glasses, 3: Funny Eyes  
    '''
    report=[]    
    if (featureType == 'SURF') and (classifierType == 'SVM'):
        results, facesList, boundingBoxes, centralCoordinates = faceDetector(I, required_size=(182, 182))
        if results != []:
            model, classLabels, standardScaler, vocabularySize, vocabulary = loadModel('SURF&SVM_K750.pkl') #'SVC_SURF_COMPLETE_750_U.pkl')
            descriptor = computeEmbedding(facesList,featureType)
            bof = bagOfFeatures(descriptor,vocabulary,vocabularySize)
            bof = standardScaler.transform(bof)
            pred = model.predict(bof)
            predArray = np.array(list(pred))
            report = reportBOF(centralCoordinates,predArray)
        else:
             print('\n\n No faces found ...\n')
             
    elif (featureType == 'SIFT') and (classifierType == 'SVM'):
        results, facesList, boundingBoxes, centralCoordinates = faceDetector(I, required_size=(182, 182))
        if results != []:
            model, classLabels, standardScaler, vocabularySize, vocabulary = loadModel('SIFT&SVM_K750.pkl')
            descriptor = computeEmbedding(facesList,featureType)
            bof = bagOfFeatures(descriptor,vocabulary,vocabularySize)
            bof = standardScaler.transform(bof)
            pred = model.predict(bof)
            predArray = np.array(list(pred))
            report = reportBOF(centralCoordinates,predArray)
        else:
             print('\n\n No faces found ...\n')
    
    elif (featureType == 'SURF') and (classifierType == 'LR'):
        results, facesList, boundingBoxes, centralCoordinates = faceDetector(I,  required_size=(182, 182))
        if results != []:
            model, classLabels, standardScaler, vocabularySize, vocabulary = loadModel('SURF&LR_K750.pkl')
            descriptor = computeEmbedding(facesList,featureType)
            bof = bagOfFeatures(descriptor,vocabulary,vocabularySize)
            bof = standardScaler.transform(bof)
            pred = model.predict(bof)
            predArray = np.array(list(pred))
            report = reportBOF(centralCoordinates,predArray)
        else:
             print('\n\n No faces found ...\n')        
        
    elif (featureType == 'SIFT') and (classifierType == 'LR'):
        results, facesList, boundingBoxes, centralCoordinates = faceDetector(I,  required_size=(182, 182))
        if results != []:
            model, classLabels, standardScaler, vocabularySize, vocabulary = loadModel('SIFT&LR_K750.pkl')
            descriptor = computeEmbedding(facesList,featureType)
            bof = bagOfFeatures(descriptor,vocabulary,vocabularySize)
            bof = standardScaler.transform(bof)
            pred = model.predict(bof)
            predArray = np.array(list(pred))
            report = reportBOF(centralCoordinates,predArray)
        else:
             print('\n\n No faces found ...\n')
             
    elif (featureType == 'NONE') and (classifierType == 'VGG16'):
        # extract the boundingBoxes, central face regions and store boundingBoxes in a pytorch tensor (faces_from_image_tensors)
        results, facesList,boundingBoxes, centralCoordinates = faceDetectorCNN(I, required_size=(224, 224))
        if results != []:
            predicted =[]
            for face in facesList:
                x = np.array(face)
                x = np.expand_dims(x, axis=0)
                x = utils.preprocess_input(x, version=1)# or version=2
                preds = VGGmodel.predict(x).argmax()
                pred = class_names[preds]
                predicted.append(pred)
            report = reportCNN(centralCoordinates,predicted)    
        else:
             print('\n\n No faces found ...\n')
  
    else:
        raise ValueError('Not a valid feature/classifier combination')
    
    im = image_exif(I)
    pixels = asarray(im)
    boxes = []
    facebox=[]
    lefteye=[]
    righteye=[]
    
    if results != []:
            print('\nApplying bounding boxes and facial landmarks ...\n') 

    
    for i, person in enumerate(results):
        x, y, width, height = person['box']
        boxes.extend([x,width,y,height])
        x1, y1 = abs(x), abs(y)
        x2, y2 = x1 + width, y1 + height
        cv2.rectangle(pixels, (x1,y1), (x2,y2), color = (0, 255, 0), thickness=3)
        cv2.rectangle(pixels, (x-3,y), ((x+width+3),(y-35)), color = (0, 255, 0), thickness=-1) 
        cv2.putText(pixels, str(report[i][0]), ((x+10), (y-5)), cv2.FONT_HERSHEY_SIMPLEX,0.8, (0, 0, 0), thickness=2) 
        for key, value in person['keypoints'].items():
            dot = Circle(value, radius=0.5, color='red')
            ax = plt.gca()
            ax.add_patch(dot)
     
        if creativeMode == '1':
            face = pixels[y1:y2, x1:x2]    
            face = cv2.blur(face, (199, 199), 0)
            pixels.setflags(write=1)
            pixels[y1:y2, x1:x2] = face
            print('Applying creative mode 1: Anonymity for person # %s successfully ... ' % (i))    
              
        if creativeMode == '2':
            glasses = cv2.imread('filters/gucciglasses2.png', cv2.IMREAD_UNCHANGED)
            facebox.append(person['box'])
            lefteye.append(person['keypoints']['left_eye'])
            righteye.append(person['keypoints']['right_eye'])
            sunglass_width=int(facebox[i][2])
            sunglass_height=int(facebox[i][3]/3)              
            
            try:
                pass
                sunglass=cv2.resize(glasses, (sunglass_width, sunglass_height))   
                overlay_mask=sunglass[:,:,3:]
                overlay_img=sunglass[:,:,:3]
                overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)
                background_mask = 255 - overlay_mask
                height,width,channels=sunglass.shape
                roi=pixels[int(righteye[i][1]-(sunglass_height/3)):int(height+righteye[i][1]-(sunglass_height/3)),int(lefteye[i][0]-(sunglass_width/4)):int(width+lefteye[i][0]-(sunglass_width/4))]
                face_part = roi * background_mask
                final_glasses = cv2.add(-face_part,overlay_img)
                pixels.setflags(write=1)           
                pixels[int(righteye[i][1]-(sunglass_height/3)):int(height+righteye[i][1]-(sunglass_height/3)),int(lefteye[i][0]-(sunglass_width/4)):int(width+lefteye[i][0]-(sunglass_width/4))]=final_glasses
                print('Applying creative mode 2: Cool glasses for person # %s successfully ... ' % (i+1))        
            except:
                print('Applying creative mode 2: Cool glasses for person # %s failed ... ' % (i+1))
                
        if creativeMode == '3':              
            eyes = cv2.imread('filters/e5.png' , cv2.IMREAD_UNCHANGED)
            facebox.append(person['box'])
            lefteye.append(person['keypoints']['left_eye'])
            righteye.append(person['keypoints']['right_eye'])
            eye_width=int(facebox[i][2])
            eye_height=int(facebox[i][3]/3)                                
           
            try:
                pass
                funnyEyes=cv2.resize(eyes, (eye_width, eye_height))
                overlay_mask=funnyEyes[:,:,3:]
                overlay_img=funnyEyes[:,:,:3]
                overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)
                background_mask = 255 - overlay_mask
                height,width,channels=funnyEyes.shape
                roi=pixels[int(righteye[i][1]-(eye_height/3)):int(height+righteye[i][1]-(eye_height/3)),int(lefteye[i][0]-(eye_width/4)):int(width+lefteye[i][0]-(eye_width/4))]
                face_part = roi * background_mask
                final_eyes = cv2.add(-face_part,overlay_img)
                pixels.setflags(write=1)           
                pixels[int(righteye[i][1]-(eye_height/3)):int(height+righteye[i][1]-(eye_height/3)),int(lefteye[i][0]-(eye_width/4)):int(width+lefteye[i][0]-(eye_width/4))]=final_eyes
                print('Applying creative mode 3: Funny eyes for person # %s successfully ... ' % (i+1))             
            except:
                print('Applying creative mode 3: Funny eyes for person # %s failed ... ' % (i+1))      
                
      
    plt.imshow(pixels)
    print('\n\nPrinting report ...')

    return report , pixels

#******************************************************************************
if __name__ == '__main__':
    if len(sys.argv) != 5:
        raise ValueError("Arguments not complete !")
    else:
        image = sys.argv[1]
        featureType = sys.argv[2]
        classifierType = sys.argv[3]
        creativeMode = sys.argv[4]

        report, pixels = RecogniseFace(image, featureType, classifierType, creativeMode)
        print('\n************ Detected faces  ************\n')
        print(report)
        print('\n******************************************\n')
        cv2.namedWindow('Detected',cv2.WINDOW_AUTOSIZE)
        imageRGB = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
        imageResized = image_resize(imageRGB, height = 800)
        cv2.imshow('Detected', imageResized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
