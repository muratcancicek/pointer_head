from .TrainingDataHandler import TrainingDataHandler
from matplotlib import pyplot as plt
from .KerasRunner import KerasRunner
from .Analyzer import Analyzer
import os, math, numpy as  np

def testSimpleML0(DataHandler, subjId = 1, tName = 'infinity'):
    handler = DataHandler(readAllDataNow = False) 
    pairs = handler.getAllHeadPoseToPointingPairs(subjId)
    print(len(pairs))
    #data, postData = handler.getHeadGazeToPointingDataFor(1, 'infinity')
    analyzer = Analyzer()
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
    name = '%s_%s_HeadGazeFilters0.pdf' % (tName, subjId)
    path = Paths.HeadGazeGraphsFolder + subjId + Paths.sep + name
    analyzer.plotHeadGazeFiltersFor(handlers, subjId, tName, path)
    #analyzer.plotHeadGazeFiltersForSubj(handlers, subjId, Paths)

def testPlottingFilters(DataHandler, subjId = 1, tName = 'infinity'):
    if isinstance(subjId, int): subjId = str(subjId)
    Paths = DataHandler.Paths
    handler = DataHandler(postDataFolder = Paths.PostDataFolders + 'PostData_pnp\\') 
    analyzer = Analyzer()
    target, headGaze = handler.getHeadGazeToPointingDataFor(subjId, tName)
    b = 0
    data = [(target[b+150:], 'Target'), (headGaze[b:], 'HeadGaze')]
    analyzer.plotHeadGazeAndPointingFo(*data,
                                       title=tName+'_'+subjId, yLim = True)
    return newPose

def testNoise(DataHandler, subjId = 1, tName = 'infinity'):
    handler = DataHandler() 
    #target, headPose = handler.getHeadPoseToPointingDataFor(subjId, tName)
    start = int(str(subjId)*3+'0')
    fakeIDs = [str(i) for i in range(start, start+10)]
    for f in fakeIDs:
        handler.saveAllFakePostDataForSubject(subjId, f)
    #handler.saveFakePostDataForSubject(subjId, fakeId, tName)
    #fakeId = '2224'
    #target, headGaze = handler.getHeadGazeToPointingDataFor(subjId, tName)
    #_, headGaze2 = handler.getHeadGazeToPointingDataFor(fakeId, tName)
    #analyzer = Analyzer()
    #analyzer.printRMSE(headGaze[:, 0], headGaze2[:, 0])
    #analyzer.printRMSE(headGaze[:, 1], headGaze2[:, 1])
    #analyzer.plotPrediction(headGaze, headGaze2, target)

def testPlottingAllSubjects(DataHandler, subjId = 1, tName = 'infinity'):
    if isinstance(subjId, int): subjId = str(subjId)
     #['1', '1111', '2', '2222', '3', '3333']  # ['1', '2', '3'] # 
    sList = os.listdir(DataHandler.Paths.PostDataFolder)
    handler = DataHandler()
    analyzer = Analyzer()
    #analyzer.plotAllSubjectsFor(handler, sList, tName)
    analyzer.saveAllSubjectsplotted(handler, sList)
    
def testKeras(DataHandler, subjId = 1, tName = 'infinity'):
    if isinstance(subjId, int): subjId = str(subjId)
    handler = DataHandler() 
    runner = KerasRunner(handler, epochs = 1, batch_size = 300)
    #data, postData = handler.getHeadPoseToPointingDataFor(subjId, tName)
    #runner.runFCNExpOnPair(data, postData)
    #runner.runFCNExpOnAllPairs(pairs)
    #runner.runFCNExpOnSubject(subjId)
    sList = [1, 2, 3] # os.listdir(DataHandler.Paths.PostDataFolder) # 
    runner.runFCNExpOnSubjectList(sList)
    
def main(DataHandler):
   #testSimpleML0(DataHandler)
   testKeras(DataHandler, 2), 'zigzag_part1_slow'
   #testCorrelation(DataHandler, 1)'random4''vertical_part1_slow''random4'
   #testPlottingFilters(DataHandler, 3, 'random4')'infinity'
   #testNoise(DataHandler, 3)
    #testPlottingAllSubjects(DataHandler, 3, 'zigzag')

if __name__ == '__main__':
    raise NotImplementedError