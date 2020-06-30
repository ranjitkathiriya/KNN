# KNN


# Language
  * Python

# Libs:
  * Numpy
  * Seaborn
  * Pandas
  * Sklearn
  
  
You should use the training and test file from the classification folder in data.zip for Part 1.
The training set contains 4000 training instances, while the test set contains 1000 test instances.
Each instance is defined by a feature vector containing 10 feature values. The first 10 columns in
both the training and test dataset correspond to the 10 features. All features are continuous
valued features (there are no categorical features)
The 11th column, in the test and training dataset, contains the class. You will notice that this is a
multi-class classification problem. There are three classes (1, 2 or 3).
The objective of part 1 of this assignment is to build a k-Nearest Neighbour (k-NN) algorithm
(you should initially set k=1). Your algorithm should take as input a training and test dataset and
will predict the appropriate class (1, 2 or 3).
The k-NN algorithm you develop for part 1 should use standard Euclidean distance for
calculating the distance between query instances and the instances in the training data. It should
report the overall accuracy of the algorithm as the percentage of test (query) instances classified
correctly.

