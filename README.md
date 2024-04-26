# ASL_NN_ECE172
# ECE 172 Fundamentals of Machine Learning Spring 2024

Dr. Hovannes Kulhandjian  
Project 2  
Due Date: 05/06/2024  

Project 2 is a continuation of Project 1. In both projects, you are using the same dataset to perform your analysis and classification.

Project 2 is based on deep neural networks. While Project 1 was based on the first half of the course, using PCA, decision trees, SVM, etc.

You should have already discussed your topic with the instructor for Project 2.

Some sample projects that will be conducted in this class that we have already discussed are:

- Handwritten digit classification
- Card face classification
- Sign language classification
- Tree health classification
- Fruit type classification
- Constellation classification (BPSK, QPSK, 8PSK, 16PSK, etc.)

## Part A: Application of Deep Convolutional Neural Network (DCNN) for Image Classification

1. Your dataset should contain at least 500 images for each category.
2. Create a DCNN architecture in MATLAB using a sample architecture shown in Figure 1.
3. Using your dataset, train your DCNN model using 80% images for training and 20% for validation.
4. Use at least 10 iterations per epoch and at least 20 epochs.
5. Adjust the learning rate and the gradient descent algorithm for optimum results.
6. Plot the accuracy vs. epochs and iterations.

7. Plot the confusion matrix. Analyze which classifier fails the most and which one performs the best.
8. Using the pre-existing models from pretrained deep neural networks in MATLAB Toolbox Deep Neural Designer, which you may need to download if you don't already have it installed:
   - [Pretrained Convolutional Neural Networks](https://www.mathworks.com/help/deeplearning/ug/pretrained-convolutional-neural-networks.html)
   Train on three different pretrained networks (select one low-end, one mid-end, and one high-end model e.g. SqueezeNet, ResNet-50, and Inception-v3) and plot the accuracy vs. epoch for each model.
   In addition, use the YOLO (You Only Look Once) pretrained algorithm; there are different versions of YOLO up to version 8.
9. Compare your results in part 6 with the four pretrained model results in part 8.
10. Which model gave you the best results?
11. Comment on the classification accuracy of each algorithm in terms of performance and the time it took to train and perform the classification.

### Figure 1: DCNN Architecture

**Useful links:**

- [Pretrained Convolutional Neural Networks](https://www.mathworks.com/help/deeplearning/ug/pretrained-convolutional-neural-networks.html)
- [Getting Started with Deep Learning Toolbox](https://www.mathworks.com/help/deeplearning/getting-started-with-deep-learning-toolbox.html)
- [Create Simple Deep Learning Network for Classification](https://www.mathworks.com/help/deeplearning/ug/create-simple-deep-learning-network-for-classification.html)
- [Plot Confusion](https://www.mathworks.com/help/deeplearning/ref/plotconfusion.html)
- [Object Detection Using YOLOv4 Deep Learning](https://www.mathworks.com/help/vision/ug/object-detection-using-yolov4-deep-learning.html)

Datasets:

- [Top 10 Open Source Datasets for Object Detection & Machine Learning in 2021](https://www.analyticsvidhya.com/blog/2021/05/top-10-open-source-datasets-for-object-detection-machine-learning-in-2021/)
