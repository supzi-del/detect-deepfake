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
- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics)

## Training the model
- Execute the .ipynb files in [detect-deepfake] on Google Colab and link the Google Drive account.
- [Trained Model (checkpint.pt)](https://drive.google.com/file/d/1jRBqpIDG7ygvyqbsRRA_8pyfz9YQNCEI/view?usp=sharing).

## Project Set-up
- Clone this project into local machine.
  
- Execute 'pip install -r requirements.txt' in terminal.
   
- In two different terminals, 'cd detect-deepfake'
- 'cd DFD'

- In 1st terminal, run Command'python server.py'
- In the 2nd one, run Command 'npm start'
  
- You've succesfully executed the Project.


Note : In the root folder(detect-deepfake), create a new folder called "models" and add the [model file](https://drive.google.com/file/d/1jRBqpIDG7ygvyqbsRRA_8pyfz9YQNCEI/view?usp=sharing) in it.

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



