from .TrainingDataHandler import TrainingDataHandler
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from .Analyzer import Analyzer
from datetime import datetime
import numpy as np
import random
import time
import math 
import os

class DLExpRunner(object):
    KERAS_FCN = 'KerasFCN' 
    TORCH_FCN = 'TorchFCN'

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
        self._expName = 'Exp'
        self._inputName = ''

    def getExpResults(self, y_test, y_hat, yList, fromDelta = False): 
        results = []
        for title, y_gd_sub in zip(self._trailsToTest, yList):
            i = y_gd_sub.shape[0]
            y_sub, y_test = y_test[1:i], y_test[i:]
            y_hat_sub, y_hat = y_hat[1:i], y_hat[i:]
            if fromDelta:
                y_sub_r, y_hat_sub_r = self._tdHandler.\
                    rebuildResultsFromDelta(y_sub, y_hat_sub, y_gd_sub[0])
            else:
                y_sub_r = self._tdHandler.unscaleData(y_sub)
                y_hat_sub_r = self._tdHandler.unscaleData(y_hat_sub)
            title = self._getFancyTrailName(title, padding = False)
            results.append((y_sub_r, y_hat_sub_r, y_gd_sub, title))
        return results
 
    def _getFancyTrailName(self, tName, padding = True): 
        words = tName.split('_')
        s = '    ' if padding else ''
        if words[-1].isdigit():
            s += '(Subject %s) %s' % (words[-1], '_'.join(words[:-1]))
        else:
            s += '%s' % tName
        return s 

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
        print()
        w = -1 #w(, )#
        hist = hist if isinstance(hist, dict) else hist.history
        t_mse, mse = hist['loss'][-1], np.square(y_test - y_hat).mean() 
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
        #self._model.summary(print_fn = lambda x: stringlist.append(x))
        #short_model_summary = "\n".join(stringlist)
        #print(short_model_summary)
        #stringlist = self._addTrailNamesToExpSummary(stringlist)
        return stringlist
       
    def _runExpOnAllPairsAsXY(self, pairs, model, fromDelta = False): 
        start = time.time()
        if fromDelta:
            getExpData = self._tdHandler.getExpDataAsDeltaFromAllPairsAsXY
        else:
            getExpData = self._tdHandler.getExpDataFromAllPairsAsXY
        expData = getExpData(pairs, 1, self._trailsToTest)
        x_train, y_train, x_test, y_test, yList = expData
        h = model.fit(x_train, y_train, self._batch_size, self._epochs)
        y_hat = model.predict(x_test) 
        results = self.getExpResults(y_test, y_hat, yList, fromDelta)
        t = time.time() - start
        summary = self.getExpSummary(expData, y_hat, h, t)
        return results, summary
        
    def chooseTrainDataGetter(self, dataKind): 
        handler = TrainingDataHandler
        self._inputName = dataKind
        if dataKind in [handler.POSE_DATA, handler.POSE_DELTA_DATA]:
            getter = self._dataHandler.getAllHeadPoseToPointingPairs
        if dataKind in [handler.ANGLE_DATA, handler.ANGLE_DELTA_DATA]:
            getter = self._dataHandler.getAllHeadAngleToPointingPairs
        return getter
        
    def _runKerasFCNExpOnAllPairsAsXY(self, pairs, fromDelta = False): 
        from .KerasRunner import KerasRunner 
        self._expName = self.KERAS_FCN
        samplePair = pairs[list(pairs.keys())[0]]
        inputD, outputD = samplePair[1].shape[-1], samplePair[0].shape[-1]
        #self._model = KerasRunner.getKerasFCNModel(inputD, outputD, 
        #                                           hiddenC = 2, hiddenD = 36,
        #                                           lr = self._lr)
        self._model = KerasRunner.getKerasLSTMModel(inputD, outputD, 
                                                   hiddenC = 2, hiddenD = 36,
                                                   lr = self._lr)
        return self._runExpOnAllPairsAsXY(pairs, self._model, fromDelta)
       
    def _runTorchFCNExpOnAllPairsAsXY(self, pairs, fromDelta = False): 
        from .TorchModel import TorchModel, TorchFCNModel
        self._expName = self.TORCH_FCN
        samplePair = pairs[list(pairs.keys())[0]]
        inputD, outputD = samplePair[1].shape[-1], samplePair[0].shape[-1]
        self._model = TorchModel(inputD, outputD, hiddenC = 2, hiddenD = 36,
                                 batch_size = self._batch_size, lr = self._lr,
                                 Model = TorchFCNModel)
        return self._runExpOnAllPairsAsXY(pairs, self._model, fromDelta)

    def runKerasFCNExpOnAllPairsAsXY(self, pairs): 
        return self._runKerasFCNExpOnAllPairsAsXY(pairs, fromDelta = False)

    def runKerasFCNDeltaExpOnAllPairsAsXY(self, pairs): 
        return self._runKerasFCNExpOnAllPairsAsXY(pairs, fromDelta = True)
      
    def runTorchFCNExpOnAllPairsAsXY(self, pairs): 
        return self._runTorchFCNExpOnAllPairsAsXY(pairs, fromDelta = False)

    def runTorchFCNDeltaExpOnAllPairsAsXY(self, pairs): 
        return self._runTorchFCNExpOnAllPairsAsXY(pairs, fromDelta = True)
          
    def chooseExpToRun(self, exp): 
        handler = TrainingDataHandler
        if self._inputName in [handler.POSE_DATA, handler.ANGLE_DATA]:
            if exp == DLExpRunner.KERAS_FCN:
                expRunner = self.runKerasFCNExpOnAllPairsAsXY
            if exp == DLExpRunner.TORCH_FCN:
                expRunner = self.runTorchFCNExpOnAllPairsAsXY
        if self._inputName in[handler.POSE_DELTA_DATA,handler.ANGLE_DELTA_DATA]:
            if exp == DLExpRunner.KERAS_FCN:
                expRunner = self.runKerasFCNDeltaExpOnAllPairsAsXY
            if exp == DLExpRunner.TORCH_FCN:
                expRunner = self.runTorchFCNDeltaExpOnAllPairsAsXY
        return expRunner
          
    def saveFCNExpResults(self, results, subjId = None): 
        results, text = results
        sep = self._dataHandler.Paths.sep
        now = str(datetime.now())[:-7].replace(':', '-').replace(' ', '_')
        if subjId:
            name = self._expName + self._inputNam + \
                'OnSubject_%s_%s.pdf' % (subjId, now)
            path = self._dataHandler.Paths.TrainingResultsFolder \
                + subjId + sep + name
        else:
            name = self._expName + self._inputName + '_%s.pdf' % (now)
            path = self._dataHandler.analysisCommonFolder + name
        self._analyzer.savePredictionResultsAsPDF(results, path, [name,'']+text)
         
    def runExpOnSubjectList(self, sList, dataType, exp): 
        sList = [str(subjId) for subjId in sList]
        pairGetter = self.chooseTrainDataGetter(dataType)
        pairs = self._tdHandler\
            .getExpDataFromAllSubjectsAsPairs(pairGetter, sList)
        self._trailsToTrain, self._trailsToTest = \
            self._dataHandler.getDefaultTestTrailsForSubjList(sList)
        expRunner = self.chooseExpToRun(exp)
        results = expRunner(pairs)
        self.saveFCNExpResults(results)