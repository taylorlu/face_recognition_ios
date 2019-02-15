# An ios app for face recognition and retrieve
1. An c++ implement of face detect, recognition and retrieve based on mtcnn, facenet and hnswlib.
2. Another web server implement based on Caffe and Flask can be found in [Facenet-Caffe](https://github.com/taylorlu/Facenet-Caffe)
# Prerequisites
1. OpenCV3.4.0
2. [NCNN](https://github.com/Tencent/ncnn) framework, use to detect face.
3. pretraned facenet model, can be downloaded in [InceptionResnet.mlmodel](https://pan.baidu.com/s/1aleEh9ceXpGisZp3V_6Xyw)
# Test platform
  App build and run on iPhone SE, arm64 architecture.
# Application
1. Enrollment and store features in local sandbox.
2. Retrieve by hnswlib.

<div align="center">
<img src="https://github.com/taylorlu/face_recognition_ios/blob/master/pics/IMG_1756.jpg" height="414" width="240" >
<img src="https://github.com/taylorlu/face_recognition_ios/blob/master/pics/IMG_1758.jpg" height="414" width="240" >
</div>

3. All the name you enrolled can be duplicated, the retrieve algorithm identify the face when the Euclidean Distance is smaller than 0.5, since all the face embedding vector is under L2 norm and confined in a high dimensional sphere whose radius is euqal to 1. The largest distance is 2.0.

<div align="center">
<img src="https://github.com/taylorlu/face_recognition_ios/blob/master/pics/centerloss.png" height="200" width="450" >
</div>
