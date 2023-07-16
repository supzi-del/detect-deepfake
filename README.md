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

1) Training and Validation Accuracy Graph:
<img width="378" alt="Accuracy Graph" src="https://github.com/supzi-del/detect-deepfake/assets/78655439/6c524a93-c3d9-4044-be58-43735f68d713">

2) Training and Validation Loss Graph:
<img width="381" alt="Loss Graph" src="https://github.com/supzi-del/detect-deepfake/assets/78655439/c8a92094-c17c-4134-a341-583dd5a5249a">

3) Confusion Matrix:
<img width="402" alt="Confusion Matrix" src="https://github.com/supzi-del/detect-deepfake/assets/78655439/0c3bbedd-1e68-40f0-9e13-8dc2979b6d56">
<br>
<p>
<ul>
<li>True positive =  63 </li>
<li>False positive =  8 </li>
<li>False negative =  12 </li>
<li>True negative =  57 </li>
</li></ul>
<p>
4) Accuracy of the Model: <br>
Calculated Accuracy 85.71428571428571

## Screenshots
<img alt="page1" src="https://github.com/supzi-del/detect-deepfake/assets/78655439/5d8b9f61-673d-4c35-bb2f-b4552764bbb4">
<img  alt="page2" src="https://github.com/supzi-del/detect-deepfake/assets/78655439/f739b1bc-7caa-4ff0-8d13-018cd75edbf0">



