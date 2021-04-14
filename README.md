# Fashion-MNIST Lab4
## 1.Introduction
The problem that will be discussed here is the problem of classification and how to make it. The data to be clasified into a given class are pictures from the Fashion-MNIST dataset. Each image belongs to one of the classes marked with the label:
Label | Description
------------ | -------------
0 | T-shirt/top
1 | Trouser
2 | Pullover
3 | Dress
4 | Coat
5 | Sandal
6 | Shirt
7 | Sneaker
8 | Bag
9 | Ankle boot

The task is to create a model that will achieve the best classification result. 

For more informations: https://github.com/zalandoresearch/fashion-mnist

## 2.Methods
###  Intro
The first thing I did was import packages and preprocess data. Preprocessing involved separating the data from the labels, followed by scaling the image data and transforming it. I have converted the list of labels so that it is a list of vectors,               
e.g. label: 2 => vector: [0,0,1,0,0,0,0,0,0,0,0], 6 = [0,0,0,0,0,0,1,0,0,0,0] etc. I also defined a simple method for displaying charts. I also split training data into training and validation data in a 4:1 ratio.

### First Classification Model
The first classification model is a Neural Network with Input Layer, Output Layer and two Hidden Layers: both are Dense Layers. Beside that there is also Flatten Layer to put data into the Dense ones.

![First Model](/images/model1.png)

### Second Classification Model
The second classification model is an extension of the first by adding two Dropout Layers after each Hidden Dense Layer. They randomly drop a certain percentage of neurons in this layer during each training epoch during the training process (in this case I set it to 0.2). They are used to regularize and avoid overfitting.

![Second Model](/images/model2.png)

### Third Classification Model
The third classification model is an extension of earlier models. This time the Convolutionary Layer was added with the Dropout Layers just behind it. Conv2D has the number of filters set to 24 and the kernel size set to 3. In addition, padding is set to the 'same' so that the data sizes at the output and input are identical.

![Third Model](/images/model3.png)
### Fourth Classification Model
The fourth classification model is also an extension of earlier models. This time a second Convolutionary Layer (number of filters set to 32, kernel size set to 3 and padding set to 'same') and Dropout Layer behind it were added. In addition, MaxPooling Layers appeared after each Conv Layer. Thanks to them, we have reduced the number of process parameters and the complexity that the network has to deal with. The pool size is set to (2, 2), which can be seen after the image data size is reduced by half.

![Fourth Model](/images/model4.png)
### Fifth Classification Model
The fifth classification model is an extension of the fourth model and it is the last one. A third Convolutional Layer (number of filters set to 64, kernel size set to 3 and padding set to 'same') was added to it in pair with Dropout Layer and BatchNormalization Layer was added just after the first Convolutional Layer. BatchNormalization Layer is scaling the outputs so that they have approximately mean 0 and standard deviation 1. I added this because I read that it can improve the results.

![Fifth Model](/images/model5.png)
## 3.Results

Loss that I use in every model is a crossentropy loss.

### First Classification Model
![First Model Acc](/images/charts/Model1AccChart.png)
![First Model Loss](/images/charts/Model1LossChart.png)

The first model clearly shows overfitting. It begins around the 15th epoch. Then we can see on the Accuracy chart how the model learns training data, while accuracy validation data practically remains unchanged. The Loss chart shows that from around 15th epoch the loss of validation data increases.

Test Results:

![First Model Result](/images/results/Model1Results.png)

### Second Classification Model
![Second Model Acc](/images/charts/Model2AccChart.png)
![Second Model Loss](/images/charts/Model2LossChart.png)

The second model shows the Dropout layer's effect on overfitting. It is still far from ideal, but the improvement compared to the first model is clearly visible. I set both models to 50 epoch because I wanted to show this difference and make it more visible.

Test Results:

![First Model Result](/images/results/Model2Results.png)
### Third Classification Model
![Third Model Acc](/images/charts/Model3AccChart.png)
![Third Model Loss](/images/charts/Model3LossChart.png)

The third model, after adding the Convolutional Layer, again fell into overfitting. The data was not be too general so the model was over learned for training data. A particularly noticeable difference between the third model and the previous one is the time of the epoch. For the previous two it was a matter of 5-10 seconds. In this case it is over a minute. However, the accuracy of test data has clearly increased.

Test Results:

![First Model Result](/images/results/Model3Results.png)
### Fourth Classification Model
![Fourth Model Acc](/images/charts/Model4AccChart.png)
![Fourth Model Loss](/images/charts/Model4LossChart.png)

The fourth model, after adding the second Convolutional Layer and MaxPooling Layers, is by far the best model among the others. The result of training data loss has halved, the result of training data accuracy has increased and the charts clearly show that both the results of loss and accuracy of training and validation data are not so far apart. Thanks to the introduction of MaxPooling Layer, the data has been more generalized, which has also led to a significant reduction in the duration of one epoch (about 30 seconds).

Test Results:

![First Model Result](/images/results/Model4Results.png)
### Fifth Classification Model
![Fifth Model Acc](/images/charts/Model5AccChart.png)
![Fifth Model Loss](/images/charts/Model5LossChart.png)

The fifth model is the most extensive model. It has 3 Convolutional Layers, MaxPooling Layers, Dropout Layers and BatchNormalization Layer. The charts show that the loss and accuracy of training and validation data are close together. In addition, the loss of test data is the smallest of all and accuracy is the highest of all (it even reached the level of acucuracy >0.93).

Test Results:

![First Model Result](/images/results/Model5Results.png)

### Results From Benchmark
Classifier | Fashion test accuracy
------------ | -------------
Fifth Classification Model | 0.9279
3 Conv+2 FC | 0.907
3 Conv+pooling+2 FC+dropout | 0.926
5 Conv+BN+pooling | 0.931
## 4.Usage
The entire program described above is in the file lab4.ipynb

### Data Entry
1.Since I use data from a .csv file in my solution, these files should be downloaded from the website https://www.kaggle.com/zalando-research/fashionmnist

2.Enter the path to the appropriate files containing test and validation data in the pd.read_csv function at the beginning of the program.
### Software Requirements
1. Python 3.7.4
2. Keras 2.3.1
3. pandas
4. numpy
5. matplotlib
