# Face Recognition (Custom Dataset) on images

<b>Language</b><br>
Python

<b>Description</b><br>
In this projectt, we will present the methods and tools used to build a face recognition pipeline to recognize faces for a classroom. In addition to using several combinations of SURF, SIFT, SVM & LR, we will use a VGG16 convolutional neural network-based architecture pre-trained on VGG-Face dataset and fine-tuned on our custom face dataset. In addition, we will use three face filters.

<b>Installation</b><br>
	1- conda create -n cv python=3.6 <br>
	2- proceed y  <br>
	3- activate cv  <br>
	4- navigate to my folder where requirments.txt is located (inside submission folder)  <br>
	5- pip install -r requirements.txt  <br>

<b>libraries</b><br>
1- MTCNN for face detection
2- TensorFlow for face recognition
3- VGG Face pretrained VGG16 model
4- Scikit-learn for face recognition (classification) with  Support Vector Machine (SVM) &  Logisitc Regression (LR)
5- SURF and SIFT features represented as bag-of-visual features

<b>Notes</b><br>
	1- This project needs a custom dataset. For privacy reason the created dataset is not included in this repository.  <br>
	2- Create a folder dataset with extracted face images of your custom dataset. <br>
	3- Split the custom dataset into train and validation. <br>
	4- Batch MTCNN & image Augmentation.IPYNB uses MTCNN to batch extract faces from images. <br>
	5- Run ML Classifier Training file and create SIFT & SURF vocabularies and put it in vocalublary folder & put the pkl files in the models folder. <br>
	6- Run Vgg16 file and put the saved models (modelVGG2.yaml & modelVGG2.h5) in the models folder. <br>
	7- Run the main file FaceRecognition file (Details in usage section) <br>

<b>Files</b><br>
	1- MTCNN folder (The face detector) <br>
	2- models folder (You need to create this folder)<br>
	3- vocabulary folder (You need to create this folder)<br>
	4- test folder (You need to create this folder) <br>
	5- filters folder (filters png images used for creative mode) <br>
	6- Batch MTCNN & image Augmentation.IPYNB <br>
	7- ML Classifier Training.IPYNB  <br>
	8- Vgg16.IPYNB <br>
	9- RecogniseFace.py (Our main function) <br>
	10- utilities.py (Required functions for the RecogniseFace main function) <br>


<b>Running Script</b><br>	
Below we are displaying the arguments and possible combinations for our function RecogniseFace.py: <br>
	Argument 1: image path <br>
	Argument 2: featureType   = (NONE, SURF, SIFT)  <br>
	Argument 3: classifierType = (SVM, LR, VGG16)  <br>
	Argument 4: creativeMode (Face Filters) = (0 = None, 1 = Anonymity , 2 = Cool Glasses, 3 = Funny eyes)  <br>

<b>Usage</b><br>
Function can be run in command line example Anaconda Prompt or from Jupyter <br>
Example running script on command line: <br>
	1- move to the script's folder (..submission\face recognition) <br>
	2- type : <br>
		python RecogniseFace.py test/1.jpg NONE VGG16 0 <br>
		python RecogniseFace.py test/1.jpg SURF SVM 1	 <br>
		python RecogniseFace.py test/1.jpg SIFT LR 2 <br>
		python RecogniseFace.py test/1.jpg NONE VGG16 3 <br>
		python RecogniseFace.py test/1.jpg SURF LR 0 <br>
		python RecogniseFace.py test/1.jpg SURF SVM 2	 <br>	
	3- output is the input image with a labelled green bounding box on faces with the predicted class. Also a report is printed with matrix P with 3 columns that includes the class predicted and face center points x & y coordinates. <br>

<b>Complete face detection and recognition pipeline</b><br>
<br><center>
<img src="https://raw.githubusercontent.com/tgalala/Full-pipeline-face-recognition-python/master/images/face.png" >
</center>
