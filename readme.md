# Online Exam Student Anti-Cheat Tool

In the age of a global pandemic the entire industry has shifted to a work from home enviornment. However Student Examintions taken online are still a tricky problem to solve as there are still a lot of loop holes for students to use while giving online exams or sitting for online lectures.


![Demo of app](https://github.com/Zorrat/Student-Online-Exam-AntiCheat-Tool/blob/master/models/images/demo.gif)

The Student Anti-Cheat Tool helps reduce these problem by detecting students faces with face recognition for identification and 
Students onscreen time with cellphone cheating detection.



Note :- Cell phone Detection uses Yolo V4 for object detection and will impact performance.
 
Requirements :
Create a virtual conda enviornment.

    conda create --name StudentAntiCheatEnv --file requirements.txt
    conda activate StudentAntiCheatEnv
Usage:
1) Place Students Face photo with their names as "Firstname Lastname.jpg"  in the "Faces/"  folder in project diretory for face recognition and identification.

2) Place the model file in correct folder i.e.
models/yolo/yolov4.h5
https://drive.google.com/file/d/1RCD4x8rudipNBahxO6Tsk4VsmhbxrmBL/view?usp=sharing

3) Press 'Q' during rendering to abort.

![Help](https://github.com/Zorrat/Student-Online-Exam-AntiCheat-Tool/blob/master/models/images/Capture.PNG)


4) Similar Usage for webcamDemo.py. This will render your webcam feed live with object detection and face detection for testing.
```
    python StudentAntiCheat.py --path "TestVideo.mp4" --name "FirstName LastName" --fps 12 --phone true --save --verbose
 ```       

   
Output:
![Output](https://github.com/Zorrat/Student-Online-Exam-AntiCheat-Tool/blob/master/models/images/verbose-final-output.PNG)



*Requires Cuda and CuDNN along with Tensorflow GPU Else CPU inference for phone detection will be extreemely slow*
```
TensorFlow version: 2.1.0
Eager execution: True
Keras version: 2.2.4-tf
Cuda version: 10.1
Cudnn version: 7.6
Num Physical GPUs Available:  1
Num Logical GPUs Available:  1
```

Note: Depending on certain Windows machines the face_recogntion library may not install correctly.

```
pip install cmake
pip install dlib
pip install face_recognition
```
If still having issue you have to build dlib with cmake.

Refrences :
YoloV4 Tensorflow by 
https://github.com/RobotEdh/Yolov-4
