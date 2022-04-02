# Medical-Prescription-Recognition

Download dataset from https://www.kaggle.com/datasets/crawford/emnist?select=emnist-balanced-train.csv and https://www.kaggle.com/datasets/crawford/emnist?select=emnist-balanced-test.csv

Three layered Neural Network which contains an input layer, one hidden layer and an output layer is used. In this approach 785 activation units (1 biased and 784 feature) are used in an input layer. In second layer of the model which is hidden layer have 25 activation units and in last layer i.e. output layer 47 activation units are considered. 

![image](https://user-images.githubusercontent.com/81741487/161369983-cd8ab715-98fe-48d5-a696-12cbd1fd3937.png)


Feedforward is performed with the training set for calculating the hypothesis and then backpropagation is done in order to reduce the error between the layers. The regularization parameter lambda is set to 0.1 to address the problem of overfitting. Optimizer is run for 800 iterations to find the best fit model.


**Result**

Training set accuracy of 94.49% 

Test set accuracy of 80.72% over test set

**Output**

![image](https://user-images.githubusercontent.com/81741487/159640552-2f3e5ed5-52ff-4bc4-afbe-7bc76bb72bbf.png)
