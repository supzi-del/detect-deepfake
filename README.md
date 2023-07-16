# detect-deepfake

## Introduction

DeepFake Video Detection is a web application that uses deep learning techniques, specifically ResNext and LSTM models, to determine whether a given video is authentic (real) or manipulated (fake). The goal of this application is to identify videos that have been altered using deepfake technology, which refers to the use of artificial intelligence to create highly realistic fake videos.

## More Information about the project

* The combination of powerful backend capabilities of **Flask and React** allowed us to build a web application that seamlessly integrates the deepfake video detection functionality with a user-friendly frontend experience.
  
- The application employs a combination of two deep learning models: ResNext and LSTM. ResNext is a convolutional neural network architecture that is commonly used for image classification tasks. It is designed to extract meaningful features from images and has been adapted for video analysis by considering frames of the video as individual images.
  
- The LSTM (Long Short-Term Memory) model is a type of recurrent neural network that can analyze sequential data, such as video frames, by capturing dependencies and patterns over time.

- The DeepFake Video Detection web application takes a video as input and processes it frame by frame. Each frame is analyzed by the ResNext model to extract relevant visual features.

- The output of the application is a prediction of whether the video is real or fake, along with a confidence ratio. The confidence ratio indicates the level of certainty the model has in its prediction.

## Dataset Used
- The Dataset we've used to train our model is [here](https://github.com/yuezunli/celeb-deepfakeforensics).

- To find our trained model follow this [link](https://drive.google.com/drive/folders/1-zErGZ9T89TplQs3ws4QVRFlqE-ljW6l?usp=sharing).

- To train our model we've took help from [here](https://github.com/abhijitjadhav1998/Deepfake_detection_using_deep_learning/tree/master/Model%20Creation).
  Thanks to them!

- To understand the project in a better way it is structured in below format:
```
DeepFake-Detection
    |
    |--- DeepFake_Detection
    |--- Implementation Video
    |--- Project-Setup.txt
    |--- Requiremnts.txt
```

## Project Set-up
To set up the project. All the steps and guidelines regarding that are listed [here](https://github.com/iamdhrutipatel/DeepFake-Detection/blob/main/Project-Setup.txt).

2. In the root folder(DeepFake_Detection), create a new folder called "model" and add the [model file](https://drive.google.com/drive/folders/1-zErGZ9T89TplQs3ws4QVRFlqE-ljW6l?usp=sharing) in it.

<b>Add these folders to the root folder(DeepFake_Detection). Since, the path has already been given to the "server.py" file and also to avoid any path related errors.</b>

## Results

1) Accuracy of the Model:
<img width="250" height="50" alt="Model Accuracy" src="https://user-images.githubusercontent.com/58872872/133935912-1def7615-6538-4c88-9134-8f94a9367965.png">

2) Training and Validation Accuracy Graph:
<img width="378" alt="Accuracy Graph" src="https://user-images.githubusercontent.com/58872872/133936040-4bfa44a7-45c5-499b-8a10-f253cbcab56c.png">

3) Training and Validation Loss Graph:
<img width="381" alt="Loss Graph" src="https://user-images.githubusercontent.com/58872872/133935983-b4d9275f-e841-4b69-86cd-79c770ea2aa1.png">

4) Confusion Matrix:
<img width="402" alt="Confusion Matrix" src="https://user-images.githubusercontent.com/58872872/133936080-d2b39804-4a99-47b8-8be4-87ba77161961.png">

## Screenshots
