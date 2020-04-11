import numpy as np
from sklearn.preprocessing import MinMaxScaler

class TrainingDataHandler(object):
    def __init__(self):
        super()
        self.x_sc = None
        self.y_sc = None
        
    def scaleData(self, x, y):
        if self.x_sc is None:
           self.x_sc = MinMaxScaler(feature_range = (0, 1))
        x_scaled = self.x_sc.fit_transform(x)
        if self.y_sc is None:
           self.y_sc = MinMaxScaler(feature_range = (0, 1))
        y_scaled = self.y_sc.fit_transform(y)
        return x_scaled, y_scaled
    
    def unscaleData(self, y_hat):
        return self.y_sc.inverse_transform(y_hat)
    
    def getPreprocessedDataFromPairs(self, pairs, preprocess): 
        return [(preprocess(postData, data)) 
                for tName, (data, postData) in pairs.items()]

    def getPreprocessedDataFromPairsAsGen(self, pairs, preprocess): 
        return ((tName, preprocess(postData, data)) 
                for tName, (data, postData) in pairs.items())

    def separateData(self, data, setRatio = 0.6): 
        s = int(setRatio * len(data))
        return data[:s], data[s:]

    def separateXY(self, x, y, setRatio = 0.6): 
        x_train, x_test = self.separateData(x, setRatio)
        y_train, y_test = self.separateData(y, setRatio)
        return x_train, y_train, x_test, y_test
    
    def getOverlappedSequences(self, x, y, length):
        y_new = y[length:]
        x_new = np.zeros((y_new.shape[0], length, x.shape[-1]))
        for i in range(y_new.shape[0]):
            x_new[i] = x[i:i+length]
        return x_new, y_new
    
    def getDeltaAsData(self, x, y):
        x_new = x[1:] - x[:-1]
        y_new = y[1:] - y[:-1]
        return x_new, y_new
    
    def rebuildDataFromDelta(self, x, y, x_bgn, y_bgn):
        x_new = np.zeros((x.shape[0] + 1, x.shape[-1]))
        y_new = np.zeros((y.shape[0] + 1, y.shape[-1]))
        x_new[0], y_new[0] = x_bgn, y_bgn
        for i in range(x.shape[0]):
            x_new[i+1] = x_new[i] + x[i]
            y_new[i+1] = y_new[i] + y[i]
        return x_new, y_new

    def rebuildResultsFromDelta(self, y, y_hat, bgn): 
        y_unsc = self.unscaleData(y)
        y_hat_unsc = self.unscaleData(y_hat)
        y_r, y_hat_r = self.rebuildDataFromDelta(y_unsc, y_hat_unsc, bgn, bgn)
        return y_r, y_hat_r
    
    def getExpDataAsDeltaFromPair(self, x, y, setRatio = 0.6):
        x_new, y_new = self.getDeltaAsData(x, y)
        x_scaled, y_scaled = self.scaleData(x_new, y_new)
        return self.separateXY(x_scaled, y_scaled, setRatio)
    
    def getExpDataAsDeltaFromAllPairs(self, pairs, setRatio = 0.6):
        def preprocess(x, y):
            x_new, y_new = self.getDeltaAsData(x, y)
            x_scaled, y_scaled = self.scaleData(x_new, y_new)
            return x_scaled, y_scaled
        data = self.getPreprocessedDataFromPairs(pairs, preprocess)
        trainData, testData = self.separateData(data, setRatio)
        return trainData, testData
    
    def mergeAllPairsAsXY(self, pairs):
        X, Y = pairs[0]
        yList = [Y]
        for x, y in pairs[1:]:
            X = np.concatenate((X, x))
            Y = np.concatenate((Y, y))
            yList.append(y)
        return X, Y, yList
    
    def getExpDataAsDeltaFromAllPairsAsXY(self, pairs, setRatio = 0.6, testSet = None):
        train, test = [], []
        for tName, (data, postData) in pairs.items():
            if not testSet is None:
                if tName in testSet:
                    test.append((postData, data))
                else:
                    train.append((postData, data))
            else:
                train.append((postData, data))
        x, y, yList = self.mergeAllPairsAsXY(train + test)
        if len(test) == 0:
            _, test = self.separateData(yList, setRatio)
        else:
            test = [y for x, y in test]
        s = sum([yy.shape[0] for yy in test])
        r = 1 - s / y.shape[0]
        expData = self.getExpDataAsDeltaFromPair(x, y, r)
        return expData + (test, )