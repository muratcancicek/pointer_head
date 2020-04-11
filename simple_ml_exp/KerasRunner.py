from .TrainingDataHandler import TrainingDataHandler
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import LSTM
from .Analyzer import Analyzer
from datetime import datetime
import numpy as np
import random
import time
import math 


class KerasRunner(object):
    def __init__(self):
        super()

    def getLSTMModel(self, x, y): 
        model = Sequential()
        model.add(LSTM(units = 50, 
                           return_sequences = True, 
                           input_shape = (x.shape[-1], 1)))
        model.add(Dropout(0.2))

        model.add(LSTM(units = 50, return_sequences = True))
        model.add(Dropout(0.2))

        model.add(LSTM(units = 50, return_sequences = True))
        model.add(Dropout(0.2))

        model.add(LSTM(units = 50))
        model.add(Dropout(0.2))

        model.add(Dense(units = y.shape[-1]))

        model.compile(optimizer = 'adam', loss = 'mean_squared_error')

        return model

    def runLSTMModel(self, x, y): 
        regressor.fit(x, y, epochs = 100, batch_size = 32)
    
    def getFCNModel(self, x, y): 
        neurons = [12, 24, 36, 48, 48, 36, 24, 12]
        model = Sequential()
        model.add(Dense(neurons[0], input_dim=x.shape[-1], activation='relu'))
        for n in neurons[1:]:
            #model.add(Dropout(0.2))
            model.add(Dense(n, activation = 'relu'))
        model.add(Dense(units = y.shape[-1]))
        model.compile(loss='mse', optimizer='adam') #, metrics=['mse'])
        return model
    
    def runFCNExpOnPair(self, expData, epochs = 50, batch_size = 10): 
        return y_hat
    
    def runFCNExpOnPair(self, data, postData, epochs = 40, batch_size = 40): 
        handler = TrainingDataHandler()
        expData = handler.getExpDataAsDeltaFromPair(postData, data)
        x_train, y_train, x_test, y_test = expData
        model = self.getFCNModel(x_train, y_train)
        model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size)
        y_hat = model.predict(x_test)
        s = int(0.6 * len(data))
        y_gd = data[s:]
        y_r, y_hat_r = handler.rebuildResultsFromDelta(y_test,y_hat, y_gd[0])
        analyzer = Analyzer()
        analyzer.plotPrediction(y_r, y_hat_r, y_gd)


    def runFCNExpOnAllPairs(self, expData, epochs=500, batch_size=10): 
        trainData, testData = expData
        model = self.getFCNModel(trainData[0][0], trainData[0][1])
        for i in range(epochs):
            print('OUTER_EPOCH %d/%d' % (i+1, epochs))
            random.shuffle(trainData)
            for x_train, y_train in trainData:
                model.fit(x_train, y_train, epochs = 1, batch_size = batch_size)
        for x_test, y_test in testData:
            loss, accuracy = model.evaluate(x_test, y_test)
            print('Loss: %.4f' % (loss))
                   
    def getExpResults(self, handler, y_test, y_hat, testSet, yList): 
        results = []
        for title, y_gd in zip(testSet, yList):
            i = y_gd.shape[0]
            y_sub, y_test = y_test[1:i], y_test[i:]
            y_hat_sub, y_hat = y_hat[1:i], y_hat[i:]
            y_r, y_hat_r = \
                handler.rebuildResultsFromDelta(y_sub, y_hat_sub, y_gd[0])
            results.append((y_r, y_hat_r, y_gd, title))
            #img = analyzer.getPredictionPlot(y_r, y_hat_r, y_gd, title, False)
        return results
         
    def getExpSummary(self, expData, y_hat, model, hist, t, epochs, batch_size): 
        x_train, y_train, x_test, y_test, yList = expData
        t_mse, mse = hist.history['loss'][-1], np.square(y_test - y_hat).mean() 
        print()
        mseText = 'Train MSE: %.4f / Test MSE: %.4f' % (t_mse, mse)
        print(mseText)
        stringlist = []
        tx, ty =  x_train.shape[1], y_train.shape[1]
        input = 'Pose' if ty == 6 else 'Gaze'
        exText = 'Head %s Change -> Target Movement' % input
        mText = '# of Input Features (%d) -> Outputs (%d)' % (tx, ty)
        tr, ts = x_train.shape[0], x_test.shape[0]
        ratText = 'Test/Train & TrainToAllRatio:' + \
            '%d/%d & %.2f' % (ts, tr, tr/(tr+ts))
        epText = 'epochs = %d, batch_size = %d' % (epochs, batch_size)
        t = ('%d sec' % t) if t < 60 else ('%d min %d sec' % (t/60, t%60))
        trText = 'Total Experiment Duration (incl. training): %s' % t
        stringlist = [exText, mText, ratText, '', 
                      epText, trText, '', mseText, '']
        model.summary(print_fn = lambda x: stringlist.append(x))
        #short_model_summary = "\n".join(stringlist)
        #print(short_model_summary)
        return stringlist
           
    def runFCNExpOnAllPairsAsXY(self, pairs, epochs = 1, batch_size = 10): 
        start = time.time()
        handler = TrainingDataHandler()
        testSet = sorted(['vertical', 'random1', 'horizontal_part1_slow'])
        expData = handler.getExpDataAsDeltaFromAllPairsAsXY(pairs, 1, testSet)
        x_train, y_train, x_test, y_test, yList = expData
        model = self.getFCNModel(x_train, y_train)
        h = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
        y_hat = model.predict(x_test) 
        results = self.getExpResults(handler, y_test, y_hat, testSet, yList)
        t = time.time() - start
        summary = self.getExpSummary(expData,y_hat,model,h,t,epochs,batch_size)
        return results, summary
             
    def runFCNExpOnSubject(self, subjId, handler, epochs = 1, batch_size = 10): 
        if isinstance(subjId, int): subjId = str(subjId)
        pairs = handler.getAllHeadPoseToPointingPairs(subjId)
        results, text = self.runFCNExpOnAllPairsAsXY(pairs, epochs, batch_size)
        analyzer = Analyzer()
        sep = handler.Paths.sep
        now = str(datetime.now())[:-7].replace(':', '-').replace(' ', '_')
        name = 'FCNExpOnSubject_%s_%s.pdf' % (subjId, now)
        path = handler.analysisFolder + subjId + sep + name
        analyzer.savePredictionResultsAsPDF(results, path, [name, ''] + text)
