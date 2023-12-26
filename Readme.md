# Tool for enhancing virtual classrooms
We have created 2 Machine-learning tools that can be used in online classrooms helping to bridge the gap between classical study techniques and the requirements of new generations (Online Classrooms). The project has been uploaded on Google Drive due to some restrictions Drive link:- https://drive.google.com/drive/folders/1jMxf01bzuXitB6nxX8hRIKtqaXoyFWjF?usp=sharing

# Emotion detection

## Introduction

This section of the project aims to classify the emotion on a person's face into one of **seven categories**, using deep convolutional neural networks. 
These classified emotions will be used to detect emotions of the student through which software will give insights of the virtual classroom that can be used by the professors to get a overview of the class.
Example :- If a students appears disengaged, the teacher could receive an alert and offer additional support or change the teaching approach. 

## Dependencies
* The model is trained on the **FER-2013** dataset which was published on International Conference on Machine Learning (ICML). This dataset consists of 35887 grayscale, 48x48 sized face images with **seven emotions** - angry, disgusted, fearful, happy, neutral, sad and surprised.
* Python 3, [OpenCV](https://opencv.org/), [Tensorflow](https://www.tensorflow.org/)


## Basic Usage
The repository is currently compatible with `tensorflow-2.0` and makes use of the Keras API using the `tensorflow.keras` library.

* First, clone the repository and enter the folder
* The folder structure is of the form:  
  src:
  * data (folder)
  * `emotions.py` (file)
  * `haarcascade_frontalface_default.xml` (file)
  * `model.h5` (file)
* open terminal in the main folder and then :- 
```bash
cd src
python emotions.py --mode display
```
* This implementation by default detects emotions on all faces in the webcam feed. With a simple 4-layer CNN, the test accuracy reached 63.2% in 50 epochs.

## Algorithm

* First, the **haar cascade** method is used to detect faces in each frame of the webcam feed.

* The region of image containing the face is resized to **48x48** and is passed as input to the CNN.

* The network outputs a list of **softmax scores** for the seven classes of emotions.

* The emotion with maximum score is displayed on the screen.

## References

* "Challenges in Representation Learning: A report on three machine learning contests." I Goodfellow, D Erhan, PL Carrier, A Courville, M Mirza, B
   Hamner, W Cukierski, Y Tang, DH Lee, Y Zhou, C Ramaiah, F Feng, R Li,  
   X Wang, D Athanasakis, J Shawe-Taylor, M Milakov, J Park, R Ionescu,
   M Popescu, C Grozea, J Bergstra, J Xie, L Romaszko, B Xu, Z Chuang, and
   Y. Bengio. arXiv 2013.


# Proctoring-Tool

This part of the Project create's an automated proctoring system where the user can be monitored automatically through the webcam and microphone. The project is divided into two parts: vision and audio based functionalities.

### Prerequisites
To run the programs, do the following:
- create a virtual environment using the command:
  - `python -m venv venv`
- activate the virtual environment
  - `cd ./venv/Scripts/activate` (windows users)
  - `source ./venv/bin/activate` (mac and linux users)
- install the requirements
  - `pip install --upgrade pip` (to upgrade pip)
  - `pip install -r requirements.txt`

Once the requirements have been installed, The programs will run successfully.
Except for the `person_and_phone.py` script which requires a model to be downloaded.

More on that later.

For vision:
```
Tensorflow>2
OpenCV
sklearn=0.19.1 # for face spoofing. 
The model used was trained with this version and does not support recent ones.
```
For audio:
```
pyaudio
speech_recognition
nltk
```

## Vision

It has six vision based functionalities right now:
1. Track eyeballs and report if candidate is looking left, right or up.
   ![eye tracking](/Proctoring-AI/gifs/1.gif)
2. Find if the candidate opens his mouth by recording the distance between lips at starting.
   ![Mouth opening detection](/Proctoring-AI/gifs/2.gif)
3. Instance segmentation to count number of people and report if no one or more than one person detected.
  ![person counting and phone detection](/Proctoring-AI/gifs/3.gif)
4. Find and report any instances of mobile phones.  
5. Head pose estimation to find where the person is looking.
   ![head pose estimation](/Proctoring-AI/gifs/4.gif)
6. Face spoofing detection
   ![face spoofing](/Proctoring-AI/gifs/5.gif)



## Audio
It is divided into two parts:
1. Audio from the microphone is recording and converted to text using Google's speech recognition API. A different thread is used to call the API such that the recording portion is not disturbed a lot, which processes the last one, appends its data to a text file and deletes it.
2. NLTK we remove the stopwods from that file. The question paper (in txt format) is taken whose stopwords are also removed and their contents are compared. Finally, the common words along with its number are presented to the proctor.

The code for this part is available in `audio_part.py`
