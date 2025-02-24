# Product-Recognition

### Assignment 1
This project implements an object recognition pipeline using traditional image processing techniques such as:
* Image denoising, since original image are corrupted by Salt-and-Pepper noise and Gaussian noise non linear filtrs are used: Median Filter followed by a Bilateral Filter.
* Feature extraction using SIFT (Scale-Invariant Feature Transform) for scaling and rotation invariance kepyoints and descriptors calculation.
* Homography matrix and perspective transformation to align the reference image to the scene image and localize object in the scene.
* Use of Template Matching for the cases where there are low keypoints descriptors matchings.


### Assignment 2
This project uses a scientific approach and ablaton studies to construct a neural network for object detection. The model has taken inspiration from VGG repetition of stages and Inception style Global Average Pooling Classifier head. In fact the spatial dimensions were averaged out before fed into the FC layer. BatchNorm had been also added to regularize the network and help convergence. Step by step, complexity to the model was added, and design choices were made, like the addition of skip connections.
The same task is than tackled through Transfer Learning and Fine Tuning of a Resnet-18.
