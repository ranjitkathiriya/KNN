import numpy as np

class Assignment(object):
# init class
    def __init__(self,n_neighbors,X_train,y_train,X_test,y_test):
        self.n_neighbors = n_neighbors
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

# calcuate distance function
    def calculaeDistance(self):
        dists = np.sqrt(np.sum(np.square(self.X_test[:, np.newaxis, :] - self.X_train), axis=2))
        return np.argsort(dists)
# calculate weights
def weightedDistance(self, calcDist, calcIndex):
    datas = []
    data = calcDist[:, :self.n_neighbors]
    index = calcIndex[:, :self.n_neighbors]
    y_val = y_train[index]
    y_val_f = np.int16(np.reshape(y_val, (len(self.y_test), self.n_neighbors)))
    data_m = 1 / data
    for data, group in zip(data_m, y_val_f):
        freq0 = 0
        freq1 = 0
        freq2 = 0
        for _data, _group in zip(data, group):
            if _group == 0:
                freq0 += _data
            if _group == 1:
                freq1 += _data
            if _group == 2:
                freq2 += _data
        datas.append((freq0, freq1, freq2))
        probablity = np.argmax(np.asarray(datas), axis=1)
    return probablity
# saccuracy
    def score(self,maxOccurances):
        score = 0
        for i in range(len(maxOccurances)):
            if maxOccurances[i] == self.y_test[i]:
                score += 1
        return score / len(self.y_test)

# minmax normalization
def MinMaxnormlizer(data):
    max_value = np.max(data,axis=0)
    min_value = np.min(data,axis=0)

    max_value

    new_value = []
    for i in data:
        data= []
        for index,j in enumerate(i):
            data.append((j - min_value[index]) / (max_value[index] - min_value[index]))
        new_value.append(data)

    new_values=np.asarray(new_value)
    return new_values
# dataset
train_data = np.genfromtxt("./data/classification/trainingData.csv", delimiter=',')
test_data = np.genfromtxt("./data/classification/testData.csv", delimiter=',')

X_train = train_data[:,0:10]
y_train = train_data[:,10:11]

X_test = test_data[:,0:10]
y_test = test_data[:,10:11]

X_train_MinMax = MinMaxnormlizer(X_train)
X_test_MinMax = MinMaxnormlizer(X_test)

ob1 = Assignment(5,X_train_MinMax,y_train,X_test_MinMax,y_test)
calcDist = ob1.calculaeDistance()
probablity = ob1.calcmaxPredicted(calcDist)
accuracy = ob1.score(probablity)
print(accuracy)