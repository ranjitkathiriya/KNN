import numpy as np

class Part1(object):

    def __init__(self,n_neighbors,X_train,y_train,X_test,y_test):
        self.n_neighbors = n_neighbors
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

# calculate ecular distance
    def calculaeDistance(self):
        dists = np.sqrt(np.sum(np.square(self.X_test[:, np.newaxis, :] - self.X_train), axis=2))
        return np.sort(dists),np.argsort(dists)
# calcuate predctions
    def calcmaxPredicted(self,distances_index):
        maxOccurances = []
        for i in distances_index[:, 0:self.n_neighbors]:
            data = []
            for j in i:
                data.append(int(self.y_train[j]))
            maxOccurances.append(np.bincount(data).argmax())
        return maxOccurances
# score
    def score(self,maxOccurances):
        score = 0
        for i in range(len(maxOccurances)):
            if maxOccurances[i] == self.y_test[i]:
                score += 1
        return score / len(self.y_test)
# take dataset
train_data = np.genfromtxt("./data/classification/trainingData.csv", delimiter=',')
test_data = np.genfromtxt("./data/classification/testData.csv", delimiter=',')


X_train = train_data[:,0:10]
y_train = train_data[:,10:11]

X_test = test_data[:,0:10]
y_test = test_data[:,10:11]

ob1 = Part1(5,X_train,y_train,X_test,y_test)

calcDist,calcIndex = ob1.calculaeDistance()
probablity = ob1.calcmaxPredicted(calcIndex)
accuracy = ob1.score(probablity)
print("Accuracy is :",accuracy*100)






