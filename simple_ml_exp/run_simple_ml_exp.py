from .TrainingDataHandler import TrainingDataHandler
from matplotlib import pyplot as plt
from .KerasRunner import KerasRunner
from .Analyzer import Analyzer
import math

def testSimpleML0(DataHandler, subjId = 1, tName = 'infinity'):
    handler = DataHandler(readAllDataNow = False) 
    pairs = handler.getAllHeadPoseToPointingPairs(subjId)
    print(len(pairs))
    #data, postData = handler.getHeadGazeToPointingDataFor(1, 'infinity')
    #analyzer = Analyzer()
    #analyzer.plotHeadGazeAndPointingFor(data, postData)

def get4Handlers(DataHandler, subjId):
    folders = DataHandler.Paths.PostDataFolderList
    fltr = lambda f: '_'.join(f.split('_')[1:])[:-1]
    handlers = [(fltr(f), DataHandler(postDataFolder = f)) for f in folders]
    return handlers
    
def testRMSEs(DataHandler, subjId = 1, tName = 'infinity'):
    if isinstance(subjId, int): subjId = str(subjId)
    handlers = get4Handlers(DataHandler, subjId)
    analyzer = Analyzer()
    rmse = analyzer.getHeadGazeFilterRMSEsFor(handlers, subjId, tName)
    #rmse = analyzer.getHeadGazeFilterRMSEsForSubj(handlers, subjId)
    print('\n'.join(rmse))
    
def testCorrelation(DataHandler, subjId = 1, tName = 'infinity'):
    if isinstance(subjId, int): subjId = str(subjId)
    handler = DataHandler() 
    analyzer = Analyzer()
    #target, headGaze = handler.getHeadGazeToPointingDataFor(1, 'infinity')
    #corr = analyzer.getHeadGazeFilterCorrelFor(target, headGaze, tName)
    pairs = handler.getAllHeadGazeToPointingPairs(subjId)
    corr = analyzer.getHeadGazeFilterCorrelForSubj(pairs)
    print('\n'.join(corr))
    
def testPlottingFilters0(DataHandler, subjId = 1, tName = 'infinity'):
    if isinstance(subjId, int): subjId = str(subjId)
    handlers = get4Handlers(DataHandler, subjId)
    Paths = DataHandler.Paths
    analyzer = Analyzer()
    name = '%s_%s_HeadGazeFilters.pdf' % (tName, subjId)
    path = Paths.HeadGazeGraphsFolder + subjId + Paths.sep + name
    analyzer.plotHeadGazeFiltersFor(handlers, subjId, tName, path)
    #analyzer.plotHeadGazeFiltersForSubj(handlers, subjId, Paths)

def testPlottingFilters(DataHandler, subjId = 1, tName = 'infinity'):
    if isinstance(subjId, int): subjId = str(subjId)
    handler = DataHandler() 
    Paths = DataHandler.Paths
    analyzer = Analyzer()
    target, headGaze = handler.getHeadGazeToPointingDataFor(subjId, tName)
    b = 150
    data = [(target[b:], 'Target'), (headGaze[b:], 'HeadGaze')]
    analyzer.plotHeadGazeAndPointingFo(*data)

def testKeras(DataHandler, subjId = 1, tName = 'infinity'):
    if isinstance(subjId, int): subjId = str(subjId)
    handler = DataHandler() 
    runner = KerasRunner(handler, epochs = 6, batch_size = 30)
    #data, postData = handler.getHeadPoseToPointingDataFor(subjId, tName)
    #runner.runFCNExpOnPair(data, postData)
    #runner.runFCNExpOnAllPairs(pairs)
    runner.runFCNExpOnSubject(subjId)
    #sList = [1, 2] # , 3
    #runner.runFCNExpOnSubjectList(sList)

     
def main(DataHandler):
   #testSimpleML0(DataHandler)
   #testKeras(DataHandler)
   testCorrelation(DataHandler, 1)
   #testPlottingFilters(DataHandler, 3, 'random4')

if __name__ == '__main__':
    raise NotImplementedError