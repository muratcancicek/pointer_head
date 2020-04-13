from .TrainingDataHandler import TrainingDataHandler
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from keras.optimizers import Adam
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
import subprocess as sp
import os

def mask_unused_gpus(leave_unmasked=1):
  ACCEPTABLE_AVAILABLE_MEMORY = 1024*5
  COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"

  try:
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    available_gpus = [i for i, x in enumerate(memory_free_values) if x > ACCEPTABLE_AVAILABLE_MEMORY]

    if len(available_gpus) < leave_unmasked: raise ValueError('Found only %d usable GPUs in the system' % len(available_gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, available_gpus[:leave_unmasked]))
  except Exception as e:
    print('"nvidia-smi" is probably not installed. GPUs are not masked', e)

#mask_unused_gpus(2)

class KerasRunner(object):
    def __init__(self, dataHandler, modelType = 'FCN', trainDataHandler = None, 
                 analyzer = None, lr = 0.001, epochs = 50, batch_size = 10):
        super()
        self._dataHandler = dataHandler
        self._trailsToTrain = []
        self._trailsToTest = []
        self._tdHandler = trainDataHandler
        if self._tdHandler is None:
            self._tdHandler = TrainingDataHandler()
        self._analyzer = analyzer
        if self._analyzer is None:
            self._analyzer = Analyzer()
        self._lr = lr
        self._epochs = epochs
        self._batch_size = batch_size 
        self._modelType = modelType
        self._model = None
        self._optimizerType = None

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
        pass

    def getFCNModel(self, x, y): 
        #neurons = [12, 24, 36, 48, 48, 36, 24, 12]
        neurons = [128]*4
        model = Sequential()
        model.add(Dense(neurons[0], input_dim=x.shape[-1], activation='relu'))
        for n in neurons[1:]:
            #model.add(Dropout(0.2))
            model.add(Dense(n, activation = 'relu'))
        model.add(Dense(units = y.shape[-1]))
        adam = Adam(lr = self._lr)
        self._optimizerType = 'Adam'
        model.compile(loss='mse', optimizer= adam)
        return model
    
    def runFCNExpOnPair(self, data, postData): 
        expData = handler.getExpDataAsDeltaFromPair(postData, data)
        x_train, y_train, x_test, y_test = expData
        self._model = self.getFCNModel(x_train, y_train)
        self._model.fit(x_train, y_train, self._batch_size, self._epochs)
        y_hat = self._model.predict(x_test)
        s = int(0.6 * len(data))
        y_gd = data[s:]
        y = self._tdHandler.rebuildResultsFromDelta(y_test, y_hat, y_gd[0])
        y = y_r, y_hat_r
        analyzer = Analyzer()
        self._analyzer.plotPrediction(y_r, y_hat_r, y_gd)

    def runFCNExpOnAllPairs(self, expData): 
        trainData, testData = expData
        self._model = self.getFCNModel(trainData[0][0], trainData[0][1])
        for i in range(epochs):
            print('OUTER_EPOCH %d/%d' % (i+1, epochs))
            random.shuffle(trainData)
            for x_train, y_train in trainData:
                self._model.fit(x_train, y_train, self._batch_size,self._epochs)
        for x_test, y_test in testData:
            loss, accuracy = self._model.evaluate(x_test, y_test)
            print('Loss: %.4f' % (loss))
         
    def _getFancyTrailName(self, tName, padding = True): 
        words = tName.split('_')
        s = '    ' if padding else ''
        if words[-1].isdigit():
            s += '(Subject %s) %s' % (words[-1], '_'.join(words[:-1]))
        else:
            s += '%s' % tName
        return s 

                   
    def getExpResults(self, y_test, y_hat, yList): 
        results = []
        for title, y_gd in zip(self._trailsToTest, yList):
            i = y_gd.shape[0]
            y_sub, y_test = y_test[1:i], y_test[i:]
            y_hat_sub, y_hat = y_hat[1:i], y_hat[i:]
            y_r, y_hat_r = \
                self._tdHandler.rebuildResultsFromDelta(y_sub,y_hat_sub,y_gd[0])
            title = self._getFancyTrailName(title, padding = False)
            results.append((y_r, y_hat_r, y_gd, title))
            #img = analyzer.getPredictionPlot(y_r, y_hat_r, y_gd, title, False)
        return results
    def _addTrailNamesToExpSummary(self, stringlist): 
        def addFrom(l):
            for tName in l:
                s = self._getFancyTrailName(tName)
                stringlist.append(s)
            stringlist.append('')
        stringlist.append('')
        stringlist.append('Trails in the Train Set:')
        addFrom(self._trailsToTrain)
        stringlist.append('Trails in the Test Set:')
        addFrom(self._trailsToTest)
        return stringlist 

    def getExpSummary(self, expData, y_hat, hist, t): 
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
        epText = 'epochs = %d, batch_size = %d'%(self._epochs, self._batch_size)
        lrText = 'Optimizer: %s, Learning Rate = %.5f' % \
            (self._modelType, self._lr)
        t = ('%d sec' % t) if t < 60 else ('%d min %d sec' % (t/60, t%60))
        trText = 'Total Experiment Duration (incl. training): %s' % t
        stringlist = [exText, mText, ratText, '', 
                      epText, lrText, trText, '', mseText, '']
        self._model.summary(print_fn = lambda x: stringlist.append(x))
        #short_model_summary = "\n".join(stringlist)
        #print(short_model_summary)
        stringlist = self._addTrailNamesToExpSummary(stringlist)
        return stringlist
             
    def saveFCNExpResults(self, results, subjId = None): 
        results, text = results
        sep = self._dataHandler.Paths.sep
        now = str(datetime.now())[:-7].replace(':', '-').replace(' ', '_')
        if subjId:
            name = 'FCNExpOnSubject_%s_%s.pdf' % (subjId, now)
            path = self._dataHandler.Paths.TrainingResultsFolder \
                + subjId + sep + name
        else:
            name = 'FCNExp_%s.pdf' % (now)
            path = self._dataHandler.analysisCommonFolder + name
        self._analyzer.savePredictionResultsAsPDF(results, path, [name,'']+text)
             
    def runFCNExpOnAllPairsAsXY(self, pairs): 
        start = time.time()
        expData = self._tdHandler.\
            getExpDataAsDeltaFromAllPairsAsXY(pairs, 1, self._trailsToTest)
        x_train, y_train, x_test, y_test, yList = expData
        self._model = self.getFCNModel(x_train, y_train)
        h = self._model.fit(x_train, y_train, self._batch_size, self._epochs)
        y_hat = self._model.predict(x_test) 
        results = self.getExpResults(y_test, y_hat, yList)
        t = time.time() - start
        summary = self.getExpSummary(expData, y_hat, h, t)
        return results, summary
           
    def runFCNExpOnSubject(self, subjId): 
        if isinstance(subjId, int): subjId = str(subjId)
        pairs = self._dataHandler.getAllHeadPoseToPointingPairs(subjId)
        self._trailsToTrain, self._trailsToTest = \
            self._dataHandler.getDefaultTestTrailsForSubj(subjId)
        results = self.runFCNExpOnAllPairsAsXY(pairs)
        self.saveFCNExpResults(results, subjId)
        
    def runFCNExpOnSubjectList(self, sList): 
        sList = [str(subjId) for subjId in sList]
        pairs = self._tdHandler\
            .getExpDataFromAllSubjectsAsPairs(self._dataHandler, sList)
        self._trailsToTrain, self._trailsToTest = \
            self._dataHandler.getDefaultTestTrailsForSubjList(sList)
        results = self.runFCNExpOnAllPairsAsXY(pairs)
        self.saveFCNExpResults(results)
