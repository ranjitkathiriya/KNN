import numpy as np

class Part1(object):

    def __init__(self,n_neighbors,X_train,y_train,X_test,y_test):
        self.n_neighbors = n_neighbors
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test


    def calculaeDistance(self):
        dists = np.sqrt(np.sum(np.square(self.X_test[:, np.newaxis, :] - self.X_train), axis=2))
        return dists

    def weightedDistance(self, calcDist, calcIndex):
        datas = []
        data = calcDist[:, :self.n_neighbors]
        index = calcIndex[:, :self.n_neighbors]
        y_val = y_train[index]
        y_val_f = np.int16(np.reshape(y_val, (len(self.y_test), self.n_neighbors)))
        data_m = 1 / np.square(data)
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

    def score(self,maxOccurances):
        score = 0
        for i in range(len(maxOccurances)):
            if maxOccurances[i] == self.y_test[i]:
                score += 1
        return score / len(self.y_test)

train_data = np.genfromtxt("./data/regression/trainingData.csv", delimiter=',')
test_data = np.genfromtxt("./data/regression/testData.csv", delimiter=',')

X_train = train_data[0:3,0:2]
y_train = train_data[0:3,10:11]

X_test = test_data[0:3,0:2]
y_test = test_data[0:3,10:11]

ob1 = Part1(1,X_train,y_train,X_test,y_test)

calcDist = ob1.calculaeDistance()



X_i = calcDist[0:,:1]

y_i = np.reshape(y_train[calcIndex][0:,:1],(1600,1))




#((1.55958863) - (-214.12071851))*2

sum_squared_residuals = []
for X,Y in zip(X_i,y_i):
#     for Xi,Yi in zip(X,Y):
    sum_squared_residuals.append((X - Y)*2)

sum_squared_res=np.asarray(sum_squared_residuals)

#sum_squared_res.shape

yaverage=np.average(y_train)

sum_squared = []
for Y in y_i:
#     for Xi,Yi in zip(X,Y):
    sum_squared.append((yaverage - Y)*2)

#len(sum_squared)

# 1 - (431.36061427 / 427.7512116610384)#

predct = 1 - (sum_squared_res / np.reshape(sum_squared,(1600,1)))

#predct.shape

#predct[:,0:1].shape