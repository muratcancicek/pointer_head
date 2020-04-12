from .TrainingDataHandler import TrainingDataHandler
from matplotlib import pyplot as plt
from .KerasRunner import KerasRunner
from .Analyzer import Analyzer
import math

def testSimpleML0(DataHandler, subjId = 1, tName = 'infinity'):
    handler = DataHandler(readAllDataNow = False) 
    #pairs = handler.getAllHeadPoseToPointingPairs(subjId)
    data, postData = handler.getHeadGazeToPointingDataFor(1, 'infinity')
    analyzer = Analyzer()
    analyzer.plotHeadGazeAndPointingFor(data, postData)


def testCorrelation(DataHandler, subjId = 1, tName = 'infinity'):
    folders = DataHandler.Paths.PostDataFolderList
    fltr = lambda f: '_'.join(f.split('_')[1:])[:-1]
    handlers = [(fltr(f), DataHandler(postDataFolder = f)) for f in folders]
    pairs = [(h.getHeadGazeToPointingDataFor(subjId, tName)[1], f)
             for f, h in handlers]
    print([f for _, f in pairs])
    analyzer = Analyzer()
    #analyzer.plotHeadGazeFiltersFor(handlers, subjId, tName)
    analyzer.plotHeadGazeFiltersForSubj(handlers, subjId)
    

def testKeras(DataHandler, subjId = 1, tName = 'infinity'):
    if isinstance(subjId, int): subjId = str(subjId)
    f = 'C:\\cStorage\\Datasets\\WhiteBallExp\\PostData_pnp_kf\\'
    handler = DataHandler(postDataFolder = f) 

    runner = KerasRunner(handler, epochs = 60, batch_size = 30)
    #data, postData = handler.getHeadPoseToPointingDataFor(subjId, tName)
    #runner.runFCNExpOnPair(data, postData)
    #runner.runFCNExpOnAllPairs(pairs)
    #runner.runFCNExpOnSubject(subjId)
    sList = [1, 2] # , 3
    runner.runFCNExpOnSubjectList(sList)

     
def main(DataHandler):
   #testSimpleML0(DataHandler)
   #testKeras(DataHandler)
   testCorrelation(DataHandler)

if __name__ == '__main__':
    raise NotImplementedError