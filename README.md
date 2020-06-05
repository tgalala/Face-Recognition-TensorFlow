# Face Recognition

<b>Language</b><br>
Python

<b>Description</b><br>
In this projectt, we will present the methods and tools used to build a face recognition pipeline to recognize faces for a classroom. In addition to using several combinations of SURF, SIFT, SVM & LR, we will use a VGG16 convolutional neural network-based architecture pre-trained on VGG-Face dataset and fine-tuned on our custom face dataset. In addition, we will use three creative modes on the detected faces.


<b>Instructions - Windows</b><br>
1- conda create -n cv python=3.6
2- proceed y
3- activate cv
4- navigate to my folder where requirments.txt is located (inside submission folder)
5- pip install -r requirements.txt


<b>Notes</b><br>
1- Environemnt ran successfully with the above steps for face recognition
2- No need to install MTCNN as it is downloaded in the folders


FOLDER STRUCTURE
**************************

A. face recognition folder
  
   1- MTCNN folder (The face detector)
   2- models folder	
   3- vocabulary folder
   4- test folder (Our test images)
   5- filters folder (filters png images used for creative mode)
   6- Batch MTCNN & image Augmentation.IPYNB
   7- ML Classifier Training.IPYNB (Also HTML version)
   8- RecogniseFace Showcase.IPYNB (Also HTML version)
   9- Vgg16.IPYNB (Also HTML version)
  10- RecogniseFace.py (Our main function)
  11- utilities.py (Required functions for the RecogniseFace main function)

B. ocr folder

    1- ocr.py
    2- frozen_east_text_detection.pb
    3- test folder containing some test images
    4- OCR Showcase (HTML format)
    5- OCR Showcase.IPNYB (Jupyter notebook)

C. TGalala Coursework CV (Main report in PDF format)
D. requirements.txt (install dendencies to run the project)
E. README.txt

	

RUNNING SCRIPTS
***********************

**  FACE RECOGNITION  **
*******************************

Options: 

Below we are displaying the arguments and possible combinations for our function RecogniseFace.py:

Argument 1: image path
Argument 2: featureType   = (NONE, SURF, SIFT) 
Argument 3: classifierType = (SVM, LR, VGG16) 
Argument 4: creativeMode = (0 = None, 1 = Anonymity , 2 = Cool Glasses, 3 = Funny eyes) 

Usage:

Function can be run in command line example Anaconda Prompt or from Jupyter (Both work perfectly)

Example running script on command line:

	1- move to the script's folder (..submission\face recognition)
	2- type :
		python RecogniseFace.py test/c9.jpg NONE VGG16 0
		python RecogniseFace.py test/c9.jpg SURF SVM 1	
		python RecogniseFace.py test/c9.jpg SIFT LR 2
		python RecogniseFace.py test/c9.jpg NONE VGG16 3
		python RecogniseFace.py test/1.jpg SURF LR 0
		python RecogniseFace.py test/3.jpg SURF SVM 2		

	3- output is the input image with a labelled green bounding box on faces with the predicted class. Also a report is printed with matrix P with 3 columns that includes the class predicted and face center points x & y coordinates.

NOTE: A Jupyter notebook and HTML version is submitted showcasing all examples for possible combinations
