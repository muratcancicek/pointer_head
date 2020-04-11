from .TrainingDataHandler import TrainingDataHandler
from matplotlib import pyplot as plt
from .KerasRunner import KerasRunner
from .Analyzer import Analyzer
import math

def testSimpleML0(DataHandler):
    handler = DataHandler(readAllDataNow = False) 
    data, postData = handler.getHeadGazeToPointingDataFor(1, 'infinity')
    analyzer = Analyzer()
    analyzer.plotHeadGazeAndPointingFor(data, postData)

def testKeras(DataHandler, subjId = 1, tName = 'infinity'):
    if isinstance(subjId, int): subjId = str(subjId)
    f = 'C:\\cStorage\\Datasets\\WhiteBallExp\\PostData\\'
    handler = DataHandler(postDataFolder = f, readAllDataNow = False) 
    runner = KerasRunner()
    #data, postData = handler.getHeadPoseToPointingDataFor(subjId, tName)
    #runner.runFCNExpOnPair(data, postData)
    #runner.runFCNExpOnAllPairs(pairs)
    runner.runFCNExpOnSubject(subjId, handler, epochs = 600, batch_size = 10)

     
def main(DataHandler):
   testSimpleML(DataHandler)
   #testKeras(DataHandler)

if __name__ == '__main__':
    raise NotImplementedError