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


def testCorrelation(DataHandler, subjId = 1, tName = 'infinity'):
    if isinstance(subjId, int): subjId = str(subjId)
    Paths = DataHandler.Paths
    folders = Paths.PostDataFolderList
    fltr = lambda f: '_'.join(f.split('_')[1:])[:-1]
    handlers = [(fltr(f), DataHandler(postDataFolder = f)) for f in folders]
    analyzer = Analyzer()
    name = '%s_%s_HeadGazeFilters.pdf' % (tName, subjId)
    path = Paths.HeadGazeGraphsFolder + subjId + Paths.sep + name
   #analyzer.plotHeadGazeFiltersFor(handlers, subjId, tName, path)
    analyzer.plotHeadGazeFiltersForSubj(handlers, subjId, Paths)
    

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
   testCorrelation(DataHandler)

if __name__ == '__main__':
    raise NotImplementedError